# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Object detection + tracking service (YOLO11 + ByteTrack) on AMD GPU.

Endpoints:
- GET  /health              : model + available class names.
- POST /detect              : boxes for a single image (Tier 1 keyframe detection).
- POST /track               : run tracking over an uploaded video, return per-track
                              summaries plus a result_id for fetching the annotated MP4.
- GET  /track/result/{id}   : stream the annotated MP4 for a completed /track call.

The video is uploaded over HTTP and the annotated result is fetched back over HTTP,
so this service shares no data volume with the app.
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional

import cv2
import torch
import torchvision
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from ultralytics import YOLO

# This ROCm torchvision build ships only a CPU NMS kernel, so run NMS on CPU while the
# model stays on the GPU.
_orig_nms = torchvision.ops.nms


def _nms_cpu(boxes, scores, iou_threshold):
    return _orig_nms(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold).to(boxes.device)


torchvision.ops.nms = _nms_cpu
torchvision.ops.boxes.nms = _nms_cpu

MODEL_NAME = os.environ.get("TRACK_MODEL", "yolo11l.pt")
TRACKER = os.environ.get("TRACK_TRACKER", "bytetrack.yaml")
CONF = float(os.environ.get("TRACK_CONF", "0.5"))
# Detector inference resolution. Small or distant objects (e.g. aerial footage) need a
# larger value such as 1280 to be detected reliably; 640 is faster and fine for close,
# street-level scenes where objects are large in frame.
IMGSZ = int(os.environ.get("TRACK_IMGSZ", "640"))
# Annotated results are held here until the app fetches them over HTTP, then deleted.
_RESULTS_DIR = os.environ.get("TRACK_RESULTS_DIR", os.path.join(tempfile.gettempdir(), "track-results"))
_RESULT_TTL_S = float(os.environ.get("TRACK_RESULT_TTL_S", "3600"))

app = FastAPI(title="object detection + tracking")


def _sweep_stale_results() -> None:
    """Drop annotated results the app never fetched, so /tmp does not grow unbounded."""
    try:
        now = time.time()
        for name in os.listdir(_RESULTS_DIR):
            fp = os.path.join(_RESULTS_DIR, name)
            if os.path.isfile(fp) and now - os.path.getmtime(fp) > _RESULT_TTL_S:
                os.unlink(fp)
    except OSError:
        # A failed sweep must not block tracking.
        pass


# The marketing name is often unset on ROCm; map the PCI device id to the model instead.
_GPU_PCI_NAMES = {
    "0x74a1": "AMD Instinct MI300X",
    "0x74a5": "AMD Instinct MI325X",
    "0x75a0": "AMD Instinct MI350X",
    "0x75a3": "AMD Instinct MI355X",
}


@lru_cache(maxsize=1)
def _gpu_name() -> str:
    name = ""
    try:
        n = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        if n and "gfx" not in n.lower():
            name = n
    except Exception:
        # Fall through to rocm-smi / rocminfo.
        pass
    if not name:
        try:
            out = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=10).stdout
            m = re.search(r"Card Model:\s*(0x[0-9a-fA-F]+)", out)
            if m and m.group(1).lower() in _GPU_PCI_NAMES:
                name = _GPU_PCI_NAMES[m.group(1).lower()]
        except Exception:
            # rocm-smi may be absent.
            pass
    if not name:
        try:
            out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10).stdout
            m = re.search(r"\bName:\s*(gfx\w+)", out)
            if m:
                name = f"AMD Instinct ({m.group(1)})"
        except Exception:
            # rocminfo may be absent.
            pass
    return name


_model: Optional[YOLO] = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_NAME)
    return _model


def _class_ids(names: Dict[int, str], wanted: List[str]) -> Optional[List[int]]:
    if not wanted:
        return None
    wl = {w.strip().lower() for w in wanted if w.strip()}
    return [i for i, n in names.items() if n.lower() in wl]


@app.get("/health")
def health() -> dict:
    m = _get_model()
    return {"status": "ok", "model": MODEL_NAME, "gpu": _gpu_name(), "classes": list(m.names.values())}


@app.post("/track")
def track(video: UploadFile = File(...), classes: str = Form("")) -> dict:
    m = _get_model()
    class_ids = _class_ids(m.names, [c for c in classes.split(",") if c.strip()])

    _sweep_stale_results()
    work = tempfile.mkdtemp()
    try:
        src = os.path.join(work, "input.mp4")
        with open(src, "wb") as f:
            f.write(video.file.read())

        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        cap.release()

        raw = os.path.join(work, "raw.mp4")
        writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        tracks: Dict[int, Dict] = defaultdict(lambda: {"label": "", "t_start": None, "t_end": None, "count": 0})

        fi = 0
        for r in m.track(
            source=src,
            stream=True,
            persist=True,
            conf=CONF,
            classes=class_ids,
            tracker=TRACKER,
            imgsz=IMGSZ,
            verbose=False,
        ):
            writer.write(r.plot())
            t = fi / fps
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().tolist()
                cls = r.boxes.cls.int().tolist()
                for tid, c in zip(ids, cls):
                    tk = tracks[tid]
                    tk["label"] = m.names[c]
                    if tk["t_start"] is None:
                        tk["t_start"] = t
                    tk["t_end"] = t
                    tk["count"] += 1
            fi += 1
        writer.release()

        os.makedirs(_RESULTS_DIR, exist_ok=True)
        result_id = uuid.uuid4().hex
        out_path = os.path.join(_RESULTS_DIR, f"{result_id}.mp4")
        # Re-encode to H.264 so the annotated video plays in the browser.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                raw,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                out_path,
            ],
            check=False,
        )
        out = [
            {"id": int(tid), "label": v["label"], "start": round(v["t_start"], 1), "end": round(v["t_end"], 1)}
            for tid, v in sorted(tracks.items())
            if v["t_start"] is not None
        ]
        return {"tracks": out, "objects": sorted({t["label"] for t in out}), "result_id": result_id}
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.get("/track/result/{result_id}")
def track_result(result_id: str) -> FileResponse:
    # Ids are generated tokens; reject any other shape and confine the path to the results dir.
    if not re.fullmatch(r"[0-9a-f]{32}", result_id):
        raise HTTPException(status_code=400, detail="invalid result id")
    base = os.path.realpath(_RESULTS_DIR)
    fp = os.path.realpath(os.path.join(base, f"{result_id}.mp4"))
    if not fp.startswith(base + os.sep) or not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="result not found")

    def _cleanup() -> None:
        try:
            os.unlink(fp)
        except OSError:
            # The stale-result sweep will catch it.
            pass

    return FileResponse(fp, media_type="video/mp4", background=BackgroundTask(_cleanup))


@app.post("/detect")
def detect(file: UploadFile = File(...), classes: str = Form("")) -> dict:
    m = _get_model()
    cids = _class_ids(m.names, [c for c in classes.split(",") if c.strip()])
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(file.file.read())
    tmp.close()
    res = m.predict(tmp.name, conf=CONF, classes=cids, imgsz=IMGSZ, verbose=False)[0]
    os.unlink(tmp.name)
    dets = [
        {
            "label": m.names[int(b.cls)],
            "conf": round(float(b.conf), 3),
            "box": [round(x, 1) for x in b.xyxy[0].tolist()],
        }
        for b in res.boxes
    ]
    return {"detections": dets}
