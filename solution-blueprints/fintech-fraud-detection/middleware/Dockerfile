# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# ---- Stage 1: Build React ----
FROM node:20-alpine AS frontend-build
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python middleware ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY middleware/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY middleware/ .

RUN mv middleware_app app

COPY --from=frontend-build /build/dist ./client_dist

EXPOSE 8080
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "app.main:app"]
