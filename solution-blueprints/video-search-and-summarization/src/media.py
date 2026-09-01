# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Media helpers: ffprobe duration, frame sampling, and Range-aware file serving."""

import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from config import DATA_DIR, FRAME_MAX_SIDE
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

_DATA_ROOT = os.path.realpath(str(DATA_DIR))


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        return max(0.0, float(out.stdout.strip()))
    except ValueError:
        return 0.0


def sample_timestamps(duration: float, n: int, start: float = 0.0, end: Optional[float] = None) -> List[float]:
    end = duration if end is None else min(end, duration)
    start = max(0.0, min(start, end))
    span = max(0.0, end - start)
    if n <= 1 or span <= 0:
        return [start]
    # Sample at the centers of n equal sub-intervals (avoids the very first/last frame).
    step = span / n
    return [round(start + step * (i + 0.5), 3) for i in range(n)]


def extract_frame_jpeg(path: Path, t: float) -> Optional[bytes]:
    src = os.path.realpath(str(path))
    if src != _DATA_ROOT and not src.startswith(_DATA_ROOT + os.sep):
        return None
    if not os.path.isfile(src):
        return None
    scale = f"scale='min({FRAME_MAX_SIDE},iw)':-2"
    # Pass the input on an inherited fd (/dev/fd/N) so no path string reaches the ffmpeg argv.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp, open(src, "rb") as fsrc:
        fd = fsrc.fileno()
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, t):.3f}",
            "-i",
            f"/dev/fd/{fd}",
            "-frames:v",
            "1",
            "-vf",
            scale,
            "-q:v",
            "3",
            tmp.name,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, pass_fds=(fd,))
        data = Path(tmp.name).read_bytes() if r.returncode == 0 else b""
    return data or None


def extract_frames(path: Path, timestamps: List[float]) -> List[bytes]:
    frames: List[bytes] = []
    for t in timestamps:
        jpg = extract_frame_jpeg(path, t)
        if jpg:
            frames.append(jpg)
    return frames


def jpeg_to_data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def serve_file_range(path: Path, request: Request, media_type: str) -> StreamingResponse:
    safe = os.path.realpath(str(path))
    if safe != _DATA_ROOT and not safe.startswith(_DATA_ROOT + os.sep):
        raise HTTPException(status_code=400, detail="invalid path")
    path = Path(safe)
    try:
        file_size = path.stat().st_size
    except OSError as e:
        raise HTTPException(status_code=404, detail="file not found") from e

    range_header = request.headers.get("range")
    start, end = 0, file_size - 1
    status = 200
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)$", range_header.strip())
        if not m:
            raise HTTPException(status_code=416, detail="invalid range")
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else file_size - 1
        if start < 0 or start >= file_size:
            raise HTTPException(status_code=416, detail="range not satisfiable")
        end = min(end, file_size - 1)
        if end < start:
            raise HTTPException(status_code=416, detail="range not satisfiable")
        status = 206
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(end - start + 1),
        }

    def _iter():
        with open(safe, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(_iter(), status_code=status, headers=headers, media_type=media_type)
