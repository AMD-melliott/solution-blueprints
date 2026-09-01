<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# UI Service — nginx Reverse Proxy + React SPA

An nginx container that serves the React TypeScript SPA and reverse-proxies API
requests to the orchestrator. This is the primary user-facing entry point.

## Routing

| Path | Target |
|------|--------|
| `/` | Static `index.html` |
| `/analyze` | `orchestrator:8003/analyze` |
| `/health` | `orchestrator:8003/health` |
| `/sys/health` | `orchestrator:8003/sys/health` |
| `/pipeline` | `orchestrator:8003/pipeline` |
| `/nginx-health` | nginx internal health check (200 OK) |

## Structure

```
ui/
├── Dockerfile           nginx:1.27-alpine; copies static build + nginx.conf
├── nginx.conf           reverse-proxy configuration
├── static/
│   └── index.html       fallback static UI (no-build version)
├── frontend/            React + TypeScript SPA (Vite)
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   ├── hooks/
│   │   └── types/
│   ├── package.json
│   └── vite.config.ts
└── mock/
    └── mock_orchestrator.py   local stub for frontend development without full stack
```

## Local Development

### With full stack (Docker Compose)

```bash
docker compose up --build -d
open http://localhost:8080
```

### Frontend dev server (hot reload)

Requires the orchestrator running on port 8003:

```bash
cd services/ui/frontend
npm install
npm run dev
```

The Vite dev server proxies `/analyze`, `/health` etc. to `http://localhost:8003`
(configured in `vite.config.ts`).

### With mock orchestrator (no backend needed)

```bash
cd services/ui
pip install fastapi uvicorn
uvicorn mock.mock_orchestrator:app --host 0.0.0.0 --port 8003
# Then run the frontend dev server
```

## Build

The `Dockerfile` runs `npm run build` and copies the `dist/` output to nginx's
`/usr/share/nginx/html/`. The build is entirely self-contained within Docker.

## Environment Variables (runtime)

The nginx container does not use environment variables at runtime. The
orchestrator upstream is configured in `nginx.conf` (Docker) or via the Helm
ConfigMap-injected `hc1.conf` (Kubernetes).
