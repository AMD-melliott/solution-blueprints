# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Centralised configuration via environment variables.
"""

from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── downstream service URLs ────────────────────────────────────────────
    llm_base_url: str = Field("http://localhost:8000", env="LLM_BASE_URL")
    llm_api_key: str = Field("", env="LLM_API_KEY")
    ner_base_url: str = Field("http://localhost:8001", env="NER_BASE_URL")
    embedding_base_url: str = Field("http://localhost:8002", env="EMBEDDING_BASE_URL")
    medcat_base_url: str = Field("http://localhost:8004", env="MEDCAT_BASE_URL")

    # ── LLM parameters ────────────────────────────────────────────────────
    llm_model: str = Field("google/medgemma-27b-it", env="LLM_MODEL")
    llm_temperature: float = Field(0.1, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(2048, env="LLM_MAX_TOKENS")
    llm_timeout: float = Field(120.0, env="LLM_TIMEOUT")

    # ── NER / embedding timeouts ──────────────────────────────────────────
    ner_timeout: float = Field(30.0, env="NER_TIMEOUT")
    embedding_timeout: float = Field(30.0, env="EMBEDDING_TIMEOUT")
    http_max_retries: int = Field(2, env="HTTP_MAX_RETRIES")

    # ── clustering / ranking thresholds ──────────────────────────────────
    entity_similarity_threshold: float = Field(0.85, env="ENTITY_SIMILARITY_THRESHOLD")
    diagnosis_min_confidence: float = Field(0.05, env="DIAGNOSIS_MIN_CONFIDENCE")

    # ── text preprocessing ────────────────────────────────────────────────
    expand_abbreviations: bool = Field(True, env="EXPAND_ABBREVIATIONS")

    # ── CORS ──────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.
    # Set to "*" for fully open (dev only).
    # When running behind nginx, the ui service origin is the only one needed.
    cors_origins: str = Field("*", env="CORS_ORIGINS")

    # ── observability ─────────────────────────────────────────────────────
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug_pipeline: bool = Field(False, env="DEBUG_PIPELINE")

    # ── service ───────────────────────────────────────────────────────────
    app_port: int = Field(8003, env="APP_PORT")

    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS env var into a list."""
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    model_config = {"env_file": ".env", "case_sensitive": False}


settings = Settings()
