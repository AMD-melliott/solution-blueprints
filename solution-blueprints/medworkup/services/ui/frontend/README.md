<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# MEDWORKUP Frontend

The web UI for the Clinical Diagnostic Assistant. It is a React + TypeScript + Vite
single-page app that calls the orchestrator's HTTP API and renders the results.

This README is written for someone connecting the backend, not necessarily a
frontend developer. You do **not** need to understand the React code to wire this
up — you only need to know which two endpoints it calls and where to point it.

---

## What the UI expects from the backend

The app makes exactly **two HTTP calls**, both as relative paths (no hardcoded
host). Something in front of it — nginx in production, the Vite dev proxy in
development — forwards those paths to the orchestrator.

| Method & path     | When                | Request body           | Response                    |
| ----------------- | ------------------- | ---------------------- | --------------------------- |
| `POST /analyze`   | User clicks Analyze | `{ "text": "<note>" }` | `AnalyzeResponse` (below)   |
| `GET /sys/health` | On load + every 30s | —                      | `SysHealthResponse` (below) |

These shapes mirror the orchestrator's `app/schemas.py` and `app/main.py`
exactly. If those change on the backend, the matching TypeScript types live in
`src/types/api.ts` and must be updated to match. **That file is the contract.**

`AnalyzeResponse` (the important one):

```jsonc
{
  "entities_raw":        [ { "text", "label", "start", "end", "source", "negated" } ],
  "entities_normalized": [ { "canonical", "label", "variants", "mention_count", "negated" } ],
  "categorized_entities": { "symptoms": [], "diseases": [], "findings": [], "procedures": [], "other": [] },
  "diagnoses":           [ { "name", "confidence", "evidence", "reasoning", "consistency_note" } ],
  "pipeline_meta":       { "stage_timings": [ { "stage", "duration_ms" } ], "total_duration_ms", "entity_count_raw", "entity_count_normalized" }
}
```

`SysHealthResponse` (drives the status pills in the header):

```jsonc
{
  "status": "ok", // or "degraded"
  "services": {
    "orchestrator": { "status": "ok", "latency_ms": 0 },
    "ner": { "status": "ok", "latency_ms": 12.3 },
    "embedding": { "status": "ok", "latency_ms": 8.1 },
    "llm": { "status": "ok", "latency_ms": 41.7 },
  },
}
```

Per-service `status` is `"ok"` / `"degraded"` / `"down"`. Errors from `/analyze`
are read as `{ "detail": ... }` or `{ "error": ... }` and shown in the error
state, so the existing FastAPI `HTTPException` shape works as-is.

---

## Running it locally

Prerequisites: **Node.js 20.19+ or 22.12+** (Vite 8 requires it). Check with
`node --version`.

```bash
cd services/ui/frontend
npm install
npm run dev          # serves on http://localhost:5173
```

With nothing else running, the app loads but the header pills stay amber
("Connecting…") and Analyze will fail — because there's no backend behind the
proxy yet. You have two ways to give it one.

### Option A — point it at the real orchestrator

The Vite dev proxy forwards `/analyze` and `/sys/health` to whatever
`VITE_PROXY_TARGET` is set to (default `http://localhost:8003`). So if the real
orchestrator is running on `:8003`, the default already works — just start the
stack and run `npm run dev`. To point elsewhere:

```bash
cp .env.example .env.local
# edit .env.local: VITE_PROXY_TARGET=http://your-host:8003
npm run dev
```

No code change is needed to switch between mock and real — only this env var.

### Option B — run against the mock (no backend / no GPU needed)

A standalone mock orchestrator lives at `services/ui/mock/mock_orchestrator.py`.
It serves the two endpoints with hand-authored, schema-accurate responses so the
whole UI can be exercised without the real services. It was built because the
real pipeline needs a GPU the frontend dev didn't have.

```bash
cd services/ui/mock
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install fastapi uvicorn
python mock_orchestrator.py        # serves on http://localhost:8003
```

Then `npm run dev` in `frontend/` in another terminal — the default proxy hits
it. The mock is **dev-only** and is not part of any production build.

---

## Things the UI fakes (read this before wiring up)

The design called for a few behaviors the backend does not currently provide.
These are implemented on the frontend as clearly-marked seams. Decide with the
team whether each should become real, stay as-is, or be removed.

1. **"Insufficient data" gating is computed client-side.** `AnalyzeResponse` has
   no sufficiency field. Before allowing Analyze, the UI runs a text heuristic
   (`src/utils/sufficiency.ts`) that looks for Vital Signs / Physical Exam /
   Labs-ECG sections and blocks the button if they're missing. This is a
   best-effort regex and **can wrongly flag a valid note**. If the backend ever
   returns a real sufficiency verdict, replace this util with that field.
   `SUFFICIENCY_BLOCKS` (top of that file) flips between hard-block and advisory
   in one line.

2. **The 9-stage progress checklist is cosmetic.** During `/analyze` the UI ticks
   through the nine stage names on a fixed timer (`src/components/StageProgress`).
   It does **not** reflect the backend's true position — `/analyze` returns one
   response at the end and does not stream progress. Real per-stage progress
   would require the backend to stream (SSE/WebSocket); until then this is
   illustrative only.

3. **The "Schedule" button does nothing.** In the Add-Note modal it logs to the
   console. There is no scheduling endpoint. Wire it to a real call or remove it.

4. **The 3 sample notes are hardcoded** in `src/utils/sampleNotes.ts` so the app
   shows content on first load. Notes added via "New" live only in memory and are
   lost on refresh — there is no persistence layer. Add one if needed.

5. **Entity character offsets are unused by the UI.** `entities_raw` includes
   `start`/`end`, shown verbatim in the Raw Entities table. The UI does not use
   them to highlight the note text (that feature was removed per design), so
   their accuracy doesn't affect rendering.

---

## Production build

```bash
npm run build        # type-checks then builds to dist/
```

`dist/` is static files (an `index.html` + hashed JS/CSS). In production these
should be served by nginx, which also proxies `/analyze` and `/sys/health` to
the orchestrator — the same relative-path setup the dev proxy mimics. The
existing `services/ui/Dockerfile` and `nginx.conf` currently serve the old
`static/index.html`; they need updating to (a) build this app and (b) serve
`dist/`. That packaging step is **not yet done** and is the natural next task.

---

## Project layout

```
services/ui/
├── frontend/                 ← this app
│   ├── src/
│   │   ├── types/api.ts       ← the API contract (mirrors schemas.py)
│   │   ├── hooks/             ← useAnalyze (POST /analyze), useSystemHealth
│   │   ├── utils/             ← sufficiency, sample notes, status helpers
│   │   └── components/        ← one folder per component (.tsx + .module.css)
│   ├── .env.example           ← proxy-target config
│   └── vite.config.ts         ← dev proxy lives here
├── mock/
│   └── mock_orchestrator.py   ← dev-only fake backend
├── static/                    ← legacy UI (to be replaced by dist/)
├── Dockerfile                 ← needs updating for the React build
└── nginx.conf                 ← proxies /analyze + /sys/health
```
