# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Backend service: ingest, index, recursive summary, hybrid Q&A, domain prompt.

Serves the single-page UI and a JSON API. Talks to ROCm vLLM (VLM + LLM) and the
BGE-M3 / open CLIP embedding services over HTTP; captions and keyframes are indexed
in ChromaDB. Storage is local disk + SQLite.
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import embed
import httpx
import store
from config import (
    DEFAULT_DOMAIN_PROMPT,
    KEYFRAMES_DIR,
    LLM_MODEL,
    TRACK_URL,
    TRACKED_DIR,
    UI_DIR,
    VIDEOS_DIR,
    VLM_MODEL,
)
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from flows import (
    analyze_video,
    answer_question,
    answer_question_stream,
    domain_prompt,
    get_analyze_progress,
    get_index_progress,
    get_tracks,
    ingest,
    set_analyze_progress,
    set_index_progress,
    track_video,
)
from media import extract_frame_jpeg, ffprobe_duration, serve_file_range
from pydantic import BaseModel

app = FastAPI(title="AMD Video Intelligence")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])


@app.on_event("startup")
def _startup() -> None:
    store.init_db()


_track_info_cache: Optional[dict] = None


def _track_info() -> dict:
    global _track_info_cache
    if _track_info_cache is None:
        try:
            with httpx.Client(timeout=4) as c:
                d = c.get(f"{TRACK_URL}/health").json()
            _track_info_cache = {"model": d.get("model") or "", "gpu": d.get("gpu") or ""}
        except Exception:
            # The tracker is optional; report it as unavailable.
            _track_info_cache = {"model": "", "gpu": ""}
    return _track_info_cache


def _auto_ingest(video: dict) -> None:
    try:
        ingest(video)
    except Exception:
        set_index_progress(video["id"], status="error")


@app.get("/api/health")
def health() -> dict:
    ti = _track_info()
    return {
        "status": "ok",
        "vlm_model": VLM_MODEL,
        "llm_model": LLM_MODEL,
        "track_model": ti["model"] or None,
        "gpu": ti["gpu"] or None,
        "videos": len(store.list_videos()),
        "domain_prompt_custom": store.get_setting("domain_prompt") is not None,
    }


class AskBody(BaseModel):
    question: str
    archive: bool = False


class AnalyzeBody(BaseModel):
    prompt: Optional[str] = None
    objects: Optional[str] = None


class DomainPromptBody(BaseModel):
    prompt: str


class ProcessingBody(BaseModel):
    chunk_seconds: float
    frames_per_chunk: int


class TrackBody(BaseModel):
    classes: List[str] = []


class RenameBody(BaseModel):
    name: str


def _safe_name(filename: str) -> str:
    base = os.path.basename(filename or "video")
    stem = base.rsplit(".", 1)[0] if "." in base else base
    return "".join(c for c in stem if c.isalnum() or c in ("_", "-")) or "video"


# Ids are server-generated hex tokens (store.new_id); reject any other shape.
_ID_RE = re.compile(r"[a-f0-9]{6,64}")


def _safe_id(video_id: str) -> str:
    if not _ID_RE.fullmatch(video_id):
        raise HTTPException(status_code=404, detail="video not found")
    return video_id


def _store_video(display_name: str, filename: str, tmp: Path) -> dict:
    size = tmp.stat().st_size
    duration = ffprobe_duration(tmp)
    if duration <= 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Not a readable video file")
    rec = store.add_video(name=display_name, filename=filename, duration=duration, size=size)
    base = os.path.realpath(str(VIDEOS_DIR))
    dest = os.path.realpath(os.path.join(base, f"{rec['id']}.mp4"))
    if not dest.startswith(base + os.sep):
        raise HTTPException(status_code=400, detail="invalid path")
    tmp.replace(dest)
    return rec


@app.post("/api/videos")
def upload_video(
    background: BackgroundTasks, file: UploadFile = File(...), name: Optional[str] = Form(None)
) -> JSONResponse:
    display_name = (name or _safe_name(file.filename or "video")).strip()
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=str(VIDEOS_DIR))
    os.close(fd)
    tmp = Path(tmp_path)
    with open(tmp, "wb") as out:
        shutil.copyfileobj(file.file, out)
    rec = _store_video(display_name, file.filename or "video.mp4", tmp)
    set_index_progress(rec["id"], phase="captioning", current=0, total=0, status="starting")
    background.add_task(_auto_ingest, rec)
    return JSONResponse(rec)


@app.get("/api/videos")
def list_videos() -> dict:
    return {"videos": store.list_videos()}


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict:
    video_id = _safe_id(video_id)
    rec = store.get_video(video_id)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    return rec


@app.patch("/api/videos/{video_id}")
def rename_video(video_id: str, body: RenameBody) -> dict:
    video_id = _safe_id(video_id)
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    rec = store.rename_video(video_id, name)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    embed.rename_video(video_id, name)
    return rec


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str) -> dict:
    video_id = _safe_id(video_id)
    embed.delete_video_index(video_id)
    if not store.delete_video(video_id):
        raise HTTPException(status_code=404, detail="video not found")
    return {"status": "deleted", "id": video_id}


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: str, request: Request) -> Response:
    video_id = _safe_id(video_id)
    rec = store.get_video(video_id)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    return serve_file_range(store.video_path(video_id), request, "video/mp4")


@app.get("/api/videos/{video_id}/thumbnail")
def thumbnail(video_id: str) -> Response:
    video_id = _safe_id(video_id)
    rec = store.get_video(video_id)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    t = min(1.0, max(0.0, float(rec["duration"]) / 2.0))
    jpg = extract_frame_jpeg(store.video_path(video_id), t)
    if not jpg:
        raise HTTPException(status_code=500, detail="could not extract thumbnail")
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/api/videos/{video_id}/keyframe/{idx}")
def keyframe(video_id: str, idx: int) -> Response:
    video_id = _safe_id(video_id)
    base = os.path.realpath(str(KEYFRAMES_DIR))
    fp = os.path.realpath(os.path.join(base, video_id, f"{int(idx)}.jpg"))
    if not fp.startswith(base + os.sep) or not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="keyframe not found")
    return Response(content=Path(fp).read_bytes(), media_type="image/jpeg")


@app.post("/api/videos/{video_id}/index")
def index_video(video_id: str) -> dict:
    video_id = _safe_id(video_id)
    rec = store.get_video(video_id)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    set_index_progress(video_id, phase="captioning", current=0, total=0, status="starting")
    try:
        chunks = ingest(rec)
    except Exception as e:
        set_index_progress(video_id, status="error")
        raise HTTPException(status_code=500, detail="indexing failed") from e
    return {"video_id": video_id, "segments": len(chunks)}


@app.get("/api/videos/{video_id}/progress")
def video_progress(video_id: str) -> dict:
    video_id = _safe_id(video_id)
    return {"index": get_index_progress(video_id), "analyze": get_analyze_progress(video_id)}


@app.post("/api/videos/{video_id}/ask")
def ask(video_id: str, body: AskBody) -> dict:
    video_id = _safe_id(video_id)
    rec = store.get_video(video_id)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    try:
        result = answer_question(rec, body.question.strip(), archive=body.archive)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Q&A failed") from e
    return {"video_id": video_id, "question": body.question, **result}


@app.post("/api/videos/{video_id}/ask/stream")
def ask_stream(video_id: str, body: AskBody) -> StreamingResponse:
    video_id = _safe_id(video_id)
    rec = store.get_video(video_id)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    def gen():
        try:
            for kind, payload in answer_question_stream(rec, body.question.strip(), archive=body.archive):
                if kind == "citations":
                    yield json.dumps({"type": "citations", "citations": payload}) + "\n"
                elif kind == "token":
                    yield json.dumps({"type": "token", "text": payload}) + "\n"
                elif kind == "done":
                    yield json.dumps({"type": "done"}) + "\n"
        except Exception:
            # Do not surface internal exception detail to the client.
            yield json.dumps({"type": "error", "message": "Q&A failed"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/videos/{video_id}/analyze")
def analyze(video_id: str, body: Optional[AnalyzeBody] = None) -> dict:
    video_id = _safe_id(video_id)
    rec = store.get_video(video_id)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    set_analyze_progress(video_id, phase="scenes", current=0, total=0, status="starting")
    instruction = body.prompt if body else None
    focus = body.objects if body else None
    try:
        result = analyze_video(rec, instruction=instruction, focus=focus)
    except Exception as e:
        set_analyze_progress(video_id, status="error")
        raise HTTPException(status_code=500, detail="analysis failed") from e
    store.replace_report(
        video_id, "analysis", json.dumps({"markdown": result["markdown"], "chunks": result.get("chunks", [])})
    )
    return {"video_id": video_id, **result}


@app.get("/api/videos/{video_id}/analysis")
def get_analysis(video_id: str) -> dict:
    video_id = _safe_id(video_id)
    rep = store.get_report(video_id, "analysis")
    if not rep:
        return {"markdown": "", "chunks": []}
    try:
        return json.loads(rep["content"])
    except (TypeError, ValueError):
        return {"markdown": rep["content"], "chunks": []}


@app.post("/api/videos/{video_id}/track")
def track(video_id: str, body: Optional[TrackBody] = None) -> dict:
    video_id = _safe_id(video_id)
    rec = store.get_video(video_id)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    classes = (body.classes if body else []) or []
    base = os.path.realpath(str(TRACKED_DIR))
    annotated_fp = os.path.realpath(os.path.join(base, f"{video_id}.mp4"))
    if not annotated_fp.startswith(base + os.sep):
        raise HTTPException(status_code=400, detail="invalid path")
    try:
        result = track_video(rec, classes)
    except Exception as e:
        raise HTTPException(status_code=500, detail="tracking failed") from e
    return {"video_id": video_id, **result, "has_annotated": os.path.exists(annotated_fp)}


@app.get("/api/videos/{video_id}/tracks")
def video_tracks(video_id: str) -> dict:
    video_id = _safe_id(video_id)
    tk = get_tracks(video_id) or {"tracks": [], "objects": [], "classes": []}
    base = os.path.realpath(str(TRACKED_DIR))
    annotated_fp = os.path.realpath(os.path.join(base, f"{video_id}.mp4"))
    tk["has_annotated"] = annotated_fp.startswith(base + os.sep) and os.path.exists(annotated_fp)
    return tk


@app.get("/api/videos/{video_id}/tracked")
def tracked_video(video_id: str, request: Request) -> Response:
    video_id = _safe_id(video_id)
    base = os.path.realpath(str(TRACKED_DIR))
    fp = os.path.realpath(os.path.join(base, f"{video_id}.mp4"))
    if not fp.startswith(base + os.sep) or not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="no tracked video")
    return serve_file_range(Path(fp), request, "video/mp4")


@app.get("/api/domain-prompt")
def get_domain_prompt() -> dict:
    return {
        "prompt": domain_prompt(),
        "is_custom": store.get_setting("domain_prompt") is not None,
        "default": DEFAULT_DOMAIN_PROMPT,
    }


@app.post("/api/domain-prompt")
def set_domain_prompt(body: DomainPromptBody) -> dict:
    store.set_setting("domain_prompt", body.prompt)
    return {"status": "ok", "prompt": body.prompt}


@app.get("/api/settings/processing")
def get_processing() -> dict:
    return store.get_processing_params()


@app.post("/api/settings/processing")
def set_processing(body: ProcessingBody) -> dict:
    cs = max(2.0, min(120.0, float(body.chunk_seconds)))
    fp = max(1, min(16, int(body.frames_per_chunk)))
    store.set_processing_params(cs, fp)
    return {"status": "ok", "chunk_seconds": cs, "frames_per_chunk": fp}


# Serve the single-page UI at "/" when present (mounted last so /api/* wins).
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
