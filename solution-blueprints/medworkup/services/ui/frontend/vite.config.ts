// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Dev proxy
 * ---------
 * The app calls /analyze and /sys/health as RELATIVE paths. In production
 * nginx proxies those to the orchestrator (see services/ui/nginx.conf). In dev
 * there is no nginx, so Vite's proxy fills the same role: it forwards those
 * paths to whatever VITE_PROXY_TARGET points at.
 *
 * Default target is the mock orchestrator (src/mock/mock_orchestrator.py) on
 * :8003 — no GPU, instant, deterministic. To hit the real orchestrator instead:
 *
 *     VITE_PROXY_TARGET=http://localhost:8003 npm run dev   # real, local
 *     VITE_PROXY_TARGET=http://some-host:8003 npm run dev   # real, remote
 *
 * (The mock also listens on 8003, so the default just works once it's running.)
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_PROXY_TARGET || "http://localhost:8003";

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/analyze": { target, changeOrigin: true },
        "/sys": { target, changeOrigin: true },
        "/pipeline": { target, changeOrigin: true },
      },
    },
  };
});
