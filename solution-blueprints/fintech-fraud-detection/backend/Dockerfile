# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-infer.txt ./
RUN pip install --no-cache-dir -r requirements-infer.txt

COPY config.yaml ./
COPY fraud_app/ ./fraud_app/

VOLUME ["/app/artifacts"]
ENV CONFIG=config.yaml
EXPOSE 8000
CMD ["uvicorn", "fraud_app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
