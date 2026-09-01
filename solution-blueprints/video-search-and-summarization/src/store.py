# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Video metadata storage: SQLite for records, local disk for files."""

import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from config import CHUNK_SECONDS, DB_PATH, FRAMES_PER_CHUNK, KEYFRAMES_DIR, REPORTS_DIR, TRACKED_DIR, VIDEOS_DIR

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                filename TEXT NOT NULL,
                duration REAL NOT NULL,
                size INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS captions (
                video_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                text TEXT NOT NULL,
                tags TEXT,
                keyframe TEXT,
                PRIMARY KEY (video_id, idx)
            )
            """
        )
        conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        for _col, _decl in (("tags", "TEXT"), ("keyframe", "TEXT")):
            try:
                conn.execute(f"ALTER TABLE captions ADD COLUMN {_col} {_decl}")
            except sqlite3.OperationalError:
                # Column already exists.
                pass
        for _col, _decl in (("chunk_seconds", "REAL"), ("frames_per_chunk", "INTEGER")):
            try:
                conn.execute(f"ALTER TABLE videos ADD COLUMN {_col} {_decl}")
            except sqlite3.OperationalError:
                # Column already exists.
                pass


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def video_path(video_id: str) -> Path:
    # Confine to VIDEOS_DIR so a crafted id cannot escape it.
    base = os.path.realpath(str(VIDEOS_DIR))
    candidate = os.path.realpath(os.path.join(base, f"{video_id}.mp4"))
    if not candidate.startswith(base + os.sep):
        raise ValueError("invalid video id")
    return Path(candidate)


def add_video(name: str, filename: str, duration: float, size: int) -> Dict:
    vid = new_id()
    rec = {
        "id": vid,
        "name": name,
        "filename": filename,
        "duration": round(duration, 2),
        "size": size,
        "created_at": time.time(),
    }
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO videos (id, name, filename, duration, size, created_at) VALUES (?,?,?,?,?,?)",
            (rec["id"], rec["name"], rec["filename"], rec["duration"], rec["size"], rec["created_at"]),
        )
    return rec


def list_videos() -> List[Dict]:
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_video(video_id: str) -> Optional[Dict]:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    return dict(row) if row else None


def get_video_by_name(name: str) -> Optional[Dict]:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM videos WHERE name = ? ORDER BY created_at DESC LIMIT 1", (name,)).fetchone()
    return dict(row) if row else None


def rename_video(video_id: str, name: str) -> Optional[Dict]:
    with _lock, _connect() as conn:
        cur = conn.execute("UPDATE videos SET name = ? WHERE id = ?", (name, video_id))
        changed = cur.rowcount
    if not changed:
        return None
    return get_video(video_id)


def delete_video(video_id: str) -> bool:
    base = os.path.realpath(str(VIDEOS_DIR))
    p = os.path.realpath(os.path.join(base, f"{video_id}.mp4"))
    if p.startswith(base + os.sep) and os.path.exists(p):
        os.unlink(p)

    # Derived artifacts: keyframes and the tracking overlay.
    try:
        kbase = os.path.realpath(str(KEYFRAMES_DIR))
        kdir = os.path.realpath(os.path.join(kbase, video_id))
        if kdir.startswith(kbase + os.sep) and os.path.isdir(kdir):
            shutil.rmtree(kdir, ignore_errors=True)

        tbase = os.path.realpath(str(TRACKED_DIR))
        tp = os.path.realpath(os.path.join(tbase, f"{video_id}.mp4"))
        if tp.startswith(tbase + os.sep) and os.path.exists(tp):
            try:
                os.unlink(tp)
            except OSError:
                # The DB rows below are what determine whether the video is gone.
                pass
    except Exception:
        # A failed cleanup must not abort the delete.
        pass

    with _lock, _connect() as conn:
        cur = conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        conn.execute("DELETE FROM reports WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM captions WHERE video_id = ?", (video_id,))
    return cur.rowcount > 0


def save_captions(video_id: str, chunks: List[Dict]) -> None:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM captions WHERE video_id = ?", (video_id,))
        conn.executemany(
            "INSERT INTO captions (video_id, idx, start, end, text, tags, keyframe) VALUES (?,?,?,?,?,?,?)",
            [
                (video_id, i, c["start"], c["end"], c["caption"], json.dumps(c.get("tags") or {}), c.get("keyframe"))
                for i, c in enumerate(chunks)
            ],
        )


def get_captions(video_id: str) -> List[Dict]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT idx, start, end, text, tags, keyframe FROM captions WHERE video_id = ? ORDER BY idx", (video_id,)
        ).fetchall()
    out: List[Dict] = []
    for r in rows:
        try:
            tags = json.loads(r["tags"]) if r["tags"] else {}
        except (TypeError, ValueError):
            tags = {}
        out.append(
            {
                "idx": r["idx"],
                "start": r["start"],
                "end": r["end"],
                "caption": r["text"],
                "tags": tags,
                "keyframe": r["keyframe"],
            }
        )
    return out


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def get_processing_params() -> Dict:
    cs = get_setting("chunk_seconds")
    fp = get_setting("frames_per_chunk")
    return {
        "chunk_seconds": float(cs) if cs else CHUNK_SECONDS,
        "frames_per_chunk": int(fp) if fp else FRAMES_PER_CHUNK,
    }


def set_processing_params(chunk_seconds: float, frames_per_chunk: int) -> None:
    set_setting("chunk_seconds", str(float(chunk_seconds)))
    set_setting("frames_per_chunk", str(int(frames_per_chunk)))


def set_video_params(video_id: str, chunk_seconds: float, frames_per_chunk: int) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE videos SET chunk_seconds = ?, frames_per_chunk = ? WHERE id = ?",
            (float(chunk_seconds), int(frames_per_chunk), video_id),
        )


def save_report(video_id: str, kind: str, content: str) -> Dict:
    rid = new_id()
    rec = {"id": rid, "video_id": video_id, "kind": kind, "content": content, "created_at": time.time()}
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO reports (id, video_id, kind, content, created_at) VALUES (?,?,?,?,?)",
            (rec["id"], rec["video_id"], rec["kind"], rec["content"], rec["created_at"]),
        )
    return rec


def list_reports(video_id: str) -> List[Dict]:
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM reports WHERE video_id = ? ORDER BY created_at DESC", (video_id,)).fetchall()
    return [dict(r) for r in rows]


def get_report(video_id: str, kind: str) -> Optional[Dict]:
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM reports WHERE video_id = ? AND kind = ? ORDER BY created_at DESC LIMIT 1",
            (video_id, kind),
        ).fetchone()
    return dict(row) if row else None


def replace_report(video_id: str, kind: str, content: str) -> Dict:
    """Keep a single current report of this kind per video (overwrites the previous one)."""
    rec = {"id": new_id(), "video_id": video_id, "kind": kind, "content": content, "created_at": time.time()}
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM reports WHERE video_id = ? AND kind = ?", (video_id, kind))
        conn.execute(
            "INSERT INTO reports (id, video_id, kind, content, created_at) VALUES (?,?,?,?,?)",
            (rec["id"], rec["video_id"], rec["kind"], rec["content"], rec["created_at"]),
        )
    return rec
