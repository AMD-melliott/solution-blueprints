# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Pipeline: ingest (chunk + keyframe + caption + tags + index), recursive summary, hybrid Q&A.

A single domain system prompt (configurable, stored in settings) cascades through
captioning, summarization, and Q&A.
"""

import json
import math
import re
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import embed
import httpx
import store
from config import (
    DEFAULT_DOMAIN_PROMPT,
    HYBRID_TOP_N,
    KEYFRAMES_DIR,
    REQUEST_TIMEOUT_S,
    RETRIEVAL_TOP_K,
    SCENE_FANIN,
    TRACK_URL,
    TRACKED_DIR,
)
from llm import llm_complete, llm_stream, vlm_caption
from media import extract_frame_jpeg, extract_frames, sample_timestamps

_ingest_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(video_id: str) -> threading.Lock:
    with _locks_guard:
        lk = _ingest_locks.get(video_id)
        if lk is None:
            lk = threading.Lock()
            _ingest_locks[video_id] = lk
        return lk


_index_progress: Dict[str, Dict] = {}
_progress_lock = threading.Lock()


def set_index_progress(video_id: str, **kw) -> None:
    with _progress_lock:
        _index_progress.setdefault(video_id, {}).update(kw)


def get_index_progress(video_id: str) -> Dict:
    with _progress_lock:
        return dict(_index_progress.get(video_id, {}))


_analyze_progress: Dict[str, Dict] = {}


def set_analyze_progress(video_id: str, **kw) -> None:
    with _progress_lock:
        _analyze_progress.setdefault(video_id, {}).update(kw)


def get_analyze_progress(video_id: str) -> Dict:
    with _progress_lock:
        return dict(_analyze_progress.get(video_id, {}))


def domain_prompt() -> str:
    return store.get_setting("domain_prompt", DEFAULT_DOMAIN_PROMPT) or DEFAULT_DOMAIN_PROMPT


def _caption_system() -> str:
    return (
        domain_prompt() + "\n\nYou are given frames sampled in chronological order from one short segment of a video. "
        'Respond with a JSON object only: {"caption": "<2-3 sentence description>", '
        '"objects": ["<distinct objects or subjects visible>"], "scene_type": "<short label>", '
        '"on_screen_text": "<any visible text, else empty>"}.'
    )


def _parse_caption(raw: str) -> Dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group(0))
            return {
                "caption": (str(d.get("caption", "")).strip() or raw.strip()),
                "tags": {
                    "objects": d.get("objects", []),
                    "scene_type": d.get("scene_type", ""),
                    "on_screen_text": d.get("on_screen_text", ""),
                },
            }
        except (ValueError, TypeError):
            pass
    return {"caption": raw.strip(), "tags": {}}


def _save_keyframe(video_id: str, idx: int, path: Path, t: float) -> Optional[str]:
    jpg = extract_frame_jpeg(path, t)
    if not jpg:
        return None
    folder = KEYFRAMES_DIR / video_id
    folder.mkdir(parents=True, exist_ok=True)
    fp = folder / f"{idx}.jpg"
    fp.write_bytes(jpg)
    return str(fp)


def _do_ingest(video: Dict) -> List[Dict]:
    """Chunk, caption with tags, extract keyframes, and index captions + keyframes."""
    path = store.video_path(video["id"])
    duration = float(video["duration"])
    params = store.get_processing_params()
    chunk_seconds = params["chunk_seconds"]
    frames_per_chunk = params["frames_per_chunk"]
    n = max(1, math.ceil(duration / chunk_seconds))
    system = _caption_system()
    set_index_progress(video["id"], phase="captioning", current=0, total=n, status="running")
    chunks: List[Dict] = []
    # Stage (idx, start, end, caption, keyframe, caption_vec, keyframe_vec) for each segment.
    # All VLM captions and embedding vectors are computed before the old index is touched so
    # that a failure leaves the existing Chroma index intact and ensure_index() keeps working.
    staged_segments: List[Tuple] = []
    try:
        for i in range(n):
            set_index_progress(video["id"], current=i)
            start = i * chunk_seconds
            end = min(duration, (i + 1) * chunk_seconds)
            if end - start < 0.5:
                continue
            frames = extract_frames(path, sample_timestamps(duration, frames_per_chunk, start, end))
            if not frames:
                continue
            keyframe = _save_keyframe(video["id"], i, path, (start + end) / 2.0)
            raw = vlm_caption(
                frames, f"Describe the segment from {start:.0f}s to {end:.0f}s.", system=system, max_tokens=400
            )
            parsed = _parse_caption(raw)
            chunks.append(
                {
                    "idx": i,
                    "start": round(start, 1),
                    "end": round(end, 1),
                    "caption": parsed["caption"],
                    "tags": parsed["tags"],
                    "keyframe": keyframe,
                }
            )
            caption_vec = embed.embed_texts([parsed["caption"]])[0]
            keyframe_vec = embed.clip_image(Path(keyframe)) if keyframe and Path(keyframe).exists() else None
            staged_segments.append((i, start, end, parsed["caption"], keyframe, caption_vec, keyframe_vec))
    except Exception:
        set_index_progress(video["id"], status="error")
        raise
    # All captions and embeddings are ready — now atomically replace the old index.
    embed.delete_video_index(video["id"])
    embed.upsert_staged_segments(video["id"], video["name"], staged_segments)
    store.save_captions(video["id"], chunks)
    store.set_video_params(video["id"], chunk_seconds, frames_per_chunk)
    set_index_progress(video["id"], current=n, status="done")
    return chunks


def ingest(video: Dict) -> List[Dict]:
    """Force (re)processing of a video, serialized per video to avoid double work."""
    with _lock_for(video["id"]):
        return _do_ingest(video)


def ensure_index(video: Dict) -> List[Dict]:
    existing = store.get_captions(video["id"])
    if existing:
        return existing
    with _lock_for(video["id"]):
        existing = store.get_captions(video["id"])
        if existing:
            return existing
        return _do_ingest(video)


def _scene_summaries(video: Dict, chunks: List[Dict]) -> List[Dict]:
    scenes: List[Dict] = []
    for s in range(0, len(chunks), SCENE_FANIN):
        group = chunks[s : s + SCENE_FANIN]
        log = "\n".join(f"[{c['start']:.0f}-{c['end']:.0f}s] {c['caption']}" for c in group)
        summary = llm_complete(
            f"Segment captions from {group[0]['start']:.0f}s to {group[-1]['end']:.0f}s:\n{log}\n\n"
            "Summarize this scene in 1-2 sentences.",
            system=domain_prompt(),
            max_tokens=300,
        )
        scenes.append({"start": group[0]["start"], "end": group[-1]["end"], "summary": summary})
        set_analyze_progress(video["id"], current=len(scenes))
    return scenes


def summarize_recursive(video: Dict, chunks: List[Dict]) -> Dict:
    scenes = _scene_summaries(video, chunks)
    scene_log = "\n".join(f"[{s['start']:.0f}-{s['end']:.0f}s] {s['summary']}" for s in scenes)
    overall = llm_complete(
        f"Scene summaries for '{video['name']}' ({float(video['duration']):.0f}s):\n{scene_log}\n\n"
        "Write one concise overall summary of the whole video in chronological order.",
        system=domain_prompt(),
        max_tokens=800,
    )
    set_analyze_progress(video["id"], phase="overall", current=len(scenes) + 1)
    return {"scenes": scenes, "summary": overall}


def analyze_video(video: Dict, instruction: Optional[str] = None, focus: Optional[str] = None) -> Dict:
    chunks = ensure_index(video)
    n_scenes = math.ceil(len(chunks) / SCENE_FANIN) if chunks else 0
    set_analyze_progress(video["id"], phase="scenes", current=0, total=n_scenes + 2, status="running")
    rec = summarize_recursive(video, chunks)
    directive = (instruction or "Summarize the video's key events in chronological order, with timestamps.").strip()
    if focus and focus.strip():
        directive += f"\nPay particular attention to: {focus.strip()}."
    timeline = "\n".join(f"[{c['start']:.0f}-{c['end']:.0f}s] {c['caption']}" for c in chunks)
    scene_log = "\n".join(f"[{s['start']:.0f}-{s['end']:.0f}s] {s['summary']}" for s in rec["scenes"])
    prompt = (
        f"Overall summary:\n{rec['summary']}\n\nScene summaries:\n{scene_log}\n\nFull timeline:\n{timeline}\n\n"
        f"Task: {directive}\n\n"
        "Format as Markdown with these sections:\n"
        "## Overview\n## Timeline  (bulleted, each line '[start-end] event')\n"
        "## Key Observations\n## Conclusion"
    )
    markdown = llm_complete(prompt, system=domain_prompt() + "\nWrite the answer in Markdown.", max_tokens=2000)
    set_analyze_progress(video["id"], phase="report", current=n_scenes + 2, status="done")
    return {
        "markdown": markdown,
        "chunks": chunks,
        "scenes": rec["scenes"],
        "summary": rec["summary"],
        "duration": float(video["duration"]),
    }


def _fuse(text_hits: List[Dict], image_hits: List[Dict], n: int) -> List[Dict]:
    """Reciprocal-rank fusion of caption-text and keyframe-image hits over segment ids."""
    scores: Dict[str, float] = {}
    info: Dict[str, Dict] = {}
    for rank, h in enumerate(text_hits):
        scores[h["seg"]] = scores.get(h["seg"], 0.0) + 1.0 / (60 + rank)
        info[h["seg"]] = h
    for rank, h in enumerate(image_hits):
        scores[h["seg"]] = scores.get(h["seg"], 0.0) + 1.0 / (60 + rank)
        info.setdefault(h["seg"], h)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [info[seg] for seg, _ in ranked]


def _prepare_answer(video: Optional[Dict], question: str, archive: bool = False) -> Dict:
    """Hybrid retrieval shared by the blocking and streaming Q&A paths."""
    scope = None if archive else (video["id"] if video else None)
    if video and not archive:
        ensure_index(video)
    text_hits = embed.query_captions(question, RETRIEVAL_TOP_K, scope)
    image_hits = embed.query_keyframes(question, RETRIEVAL_TOP_K, scope)
    fused = _fuse(text_hits, image_hits, HYBRID_TOP_N)
    tk = get_tracks(video["id"]) if (video and not archive) else None
    track_ctx = ""
    if tk and tk.get("tracks"):
        track_ctx = "Tracked objects in this video: " + "; ".join(
            f"{t['label']} (id {t['id']}, {t['start']:.0f}-{t['end']:.0f}s)" for t in tk["tracks"][:40]
        )
    if not fused and not track_ctx:
        return {
            "citations": [],
            "prompt": None,
            "system": None,
            "fallback": "I don't know - there are no indexed segments to answer from.",
        }
    ctx_lines: List[str] = []
    citations: List[Dict] = []
    if track_ctx:
        ctx_lines.append(track_ctx)
    for h in fused:
        m = h["meta"]
        ctx_lines.append(f"({m['video_name']}, seg{m['segment_id']}, {m['start']:.0f}-{m['end']:.0f}s): {h['doc']}")
        citations.append(
            {
                "video": m["video_name"],
                "video_id": m["video_id"],
                "segment_id": m["segment_id"],
                "start": m["start"],
                "end": m["end"],
            }
        )
    system = domain_prompt() + (
        "\n\nAnswer ONLY using the retrieved segments below. Cite the segments you rely on as "
        "(video, segN, start-end seconds). If the segments do not contain the answer, say you do not know."
    )
    prompt = (
        "Retrieved video segments:\n" + "\n".join(ctx_lines) + f"\n\nQuestion: {question}\n\nAnswer with citations."
    )
    return {"citations": citations, "prompt": prompt, "system": system, "fallback": None}


def answer_question(video: Optional[Dict], question: str, archive: bool = False) -> Dict:
    prep = _prepare_answer(video, question, archive)
    if prep["fallback"] is not None:
        return {"answer": prep["fallback"], "citations": []}
    answer = llm_complete(prep["prompt"], system=prep["system"], max_tokens=1024)
    return {"answer": answer, "citations": prep["citations"]}


def _visible_after_think(raw: str) -> str:
    """Drop a Qwen3 <think> block while streaming, tolerating tags split across chunks."""
    close = raw.find("</think>")
    if close != -1:
        return raw[close + len("</think>") :].lstrip("\n")
    if "<think>" in raw:
        return ""
    if raw and "<think>".startswith(raw):
        return ""
    return raw


def answer_question_stream(video: Optional[Dict], question: str, archive: bool = False) -> Iterator[Tuple[str, object]]:
    """Yield ('citations', list), then ('token', str) deltas, then ('done', None)."""
    prep = _prepare_answer(video, question, archive)
    yield ("citations", prep["citations"])
    if prep["fallback"] is not None:
        yield ("token", prep["fallback"])
        yield ("done", None)
        return
    raw = ""
    emitted = 0
    for piece in llm_stream(prep["prompt"], system=prep["system"], max_tokens=1024):
        raw += piece
        visible = _visible_after_think(raw)
        if len(visible) > emitted:
            yield ("token", visible[emitted:])
            emitted = len(visible)
    yield ("done", None)


def track_video(video: Dict, classes: List[str]) -> Dict:
    """Run detection + tracking on the video, save the annotated MP4 and the tracks.

    The video is uploaded to the tracker and the annotated result is fetched back over
    HTTP, so the two services share no filesystem.
    """
    vid = video["id"]
    src = store.video_path(vid)
    out_path = TRACKED_DIR / f"{vid}.mp4"
    TRACKED_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as c:
        with open(src, "rb") as fh:
            r = c.post(
                f"{TRACK_URL}/track",
                files={"video": (f"{vid}.mp4", fh, "video/mp4")},
                data={"classes": ",".join(classes)},
            )
        r.raise_for_status()
        data = r.json()
        result_id = data.get("result_id")
        if result_id:
            with c.stream("GET", f"{TRACK_URL}/track/result/{result_id}") as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as out:
                    for chunk in resp.iter_bytes():
                        out.write(chunk)
    store.replace_report(
        vid,
        "tracks",
        json.dumps({"tracks": data.get("tracks", []), "objects": data.get("objects", []), "classes": classes}),
    )
    return {"tracks": data.get("tracks", []), "objects": data.get("objects", [])}


def get_tracks(video_id: str) -> Optional[Dict]:
    rep = store.get_report(video_id, "tracks")
    if not rep:
        return None
    try:
        return json.loads(rep["content"])
    except (TypeError, ValueError):
        return None
