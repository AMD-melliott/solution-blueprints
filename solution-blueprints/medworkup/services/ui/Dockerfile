# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# ─────────────────────────────────────────────
# HC-1 UI Service
# Stage 1 – build the React/Vite frontend
# Stage 2 – serve with nginx + reverse proxy
# ─────────────────────────────────────────────

# ── build stage ──────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

COPY frontend/ ./
RUN npm run build

# ── runtime stage ────────────────────────────
FROM nginx:1.27-alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Inject our config
COPY nginx.conf /etc/nginx/conf.d/hc1.conf

# Copy compiled assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Non-root: nginx master still needs 0 to bind :80,
# but worker processes drop to 'nginx' user automatically.
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:8080/nginx-health || exit 1

CMD ["nginx", "-g", "daemon off;"]
