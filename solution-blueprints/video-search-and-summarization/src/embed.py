# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Embedding clients (BGE-M3 text, CLIP image/text) and the ChromaDB index.

Captions are embedded with BGE-M3 and stored in the "captions" collection; keyframe
images are embedded with CLIP and stored in the "keyframes" collection. Both use cosine
space. Text queries hit captions directly and keyframes via CLIP's text encoder (shared
image/text space), which is what makes hybrid retrieval possible.

Both embedders are vLLM servers (aimchart-embedding) exposing an OpenAI-compatible
/v1/embeddings endpoint. Text is sent in the plain "input" form; keyframe images use the
chat-style "messages" form with an image_url, which routes to CLIP's image encoder.
"""

import base64
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import chromadb
import httpx
from config import (
    CHROMA_DIR,
    CHROMA_URL,
    EMBED_IMAGE_MODEL,
    EMBED_IMAGE_URL,
    EMBED_TEXT_MODEL,
    EMBED_TEXT_URL,
    REQUEST_TIMEOUT_S,
)

_client = None
_captions = None
_keyframes = None


def _collections():
    global _client, _captions, _keyframes
    if _client is None:
        if CHROMA_URL:
            u = urlparse(CHROMA_URL if "://" in CHROMA_URL else "http://" + CHROMA_URL)
            _client = chromadb.HttpClient(host=u.hostname, port=u.port or 8000)
        else:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _captions = _client.get_or_create_collection("captions", metadata={"hnsw:space": "cosine"})
        _keyframes = _client.get_or_create_collection("keyframes", metadata={"hnsw:space": "cosine"})
    return _captions, _keyframes


def _embed_text(base_url: str, model: str, texts: List[str]) -> List[List[float]]:
    """Plain OpenAI-style text embeddings (vLLM /v1/embeddings)."""
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as c:
        r = c.post(f"{base_url}/v1/embeddings", json={"model": model, "input": texts, "encoding_format": "float"})
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]


def embed_texts(texts: List[str]) -> List[List[float]]:
    return _embed_text(EMBED_TEXT_URL, EMBED_TEXT_MODEL, texts)


def clip_texts(texts: List[str]) -> List[List[float]]:
    return _embed_text(EMBED_IMAGE_URL, EMBED_IMAGE_MODEL, texts)


def clip_image(path: Path) -> List[float]:
    """Embed a keyframe image via the chat-style /v1/embeddings request (CLIP image encoder)."""
    with open(path, "rb") as f:
        data_uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
    body = {
        "model": EMBED_IMAGE_MODEL,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}]}],
        "encoding_format": "float",
    }
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as c:
        r = c.post(f"{EMBED_IMAGE_URL}/v1/embeddings", json=body)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]


def index_segment(
    video_id: str, idx: int, start: float, end: float, video_name: str, caption_text: str, keyframe_path: Optional[str]
) -> None:
    captions, keyframes = _collections()
    seg_id = f"{video_id}:{idx}"
    meta = {
        "video_id": video_id,
        "video_name": video_name,
        "segment_id": idx,
        "start": float(start),
        "end": float(end),
    }
    captions.upsert(
        ids=[seg_id], embeddings=[embed_texts([caption_text])[0]], documents=[caption_text], metadatas=[meta]
    )
    if keyframe_path and Path(keyframe_path).exists():
        keyframes.upsert(
            ids=[seg_id], embeddings=[clip_image(Path(keyframe_path))], documents=[caption_text], metadatas=[meta]
        )


def delete_video_index(video_id: str) -> None:
    captions, keyframes = _collections()
    for col in (captions, keyframes):
        try:
            col.delete(where={"video_id": video_id})
        except Exception:
            # Nothing indexed for this video yet.
            pass


def upsert_staged_segments(video_id: str, video_name: str, staged: list) -> None:
    """Upsert pre-computed embedding vectors for a batch of segments.

    Each entry in *staged* must be a tuple of:
        (idx, start, end, caption_text, keyframe_path_or_none, caption_vec, keyframe_vec_or_none)
    Callers are expected to have already deleted the old index for *video_id* before calling this.
    """
    captions, keyframes = _collections()
    for idx, start, end, caption_text, keyframe_path, caption_vec, keyframe_vec in staged:
        seg_id = f"{video_id}:{idx}"
        meta = {
            "video_id": video_id,
            "video_name": video_name,
            "segment_id": idx,
            "start": float(start),
            "end": float(end),
        }
        captions.upsert(ids=[seg_id], embeddings=[caption_vec], documents=[caption_text], metadatas=[meta])
        if keyframe_vec is not None:
            keyframes.upsert(ids=[seg_id], embeddings=[keyframe_vec], documents=[caption_text], metadatas=[meta])


def rename_video(video_id: str, new_name: str) -> None:
    """Keep citation metadata in sync when a video is renamed."""
    captions, keyframes = _collections()
    for col in (captions, keyframes):
        try:
            got = col.get(where={"video_id": video_id})
            ids = got.get("ids", [])
            metas = got.get("metadatas", []) or []
            if ids:
                for m in metas:
                    m["video_name"] = new_name
                col.update(ids=ids, metadatas=metas)
        except Exception:
            # Collection may not exist yet; citations fall back to the stored name.
            pass


def _query(col, qvec: List[float], k: int, video_id: Optional[str]) -> List[Dict]:
    res = col.query(query_embeddings=[qvec], n_results=k, where=({"video_id": video_id} if video_id else None))
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]
    return [{"seg": ids[i], "distance": dists[i], "meta": metas[i], "doc": docs[i]} for i in range(len(ids))]


def query_captions(query_text: str, k: int, video_id: Optional[str] = None) -> List[Dict]:
    captions, _ = _collections()
    return _query(captions, embed_texts([query_text])[0], k, video_id)


def query_keyframes(query_text: str, k: int, video_id: Optional[str] = None) -> List[Dict]:
    _, keyframes = _collections()
    return _query(keyframes, clip_texts([query_text])[0], k, video_id)
