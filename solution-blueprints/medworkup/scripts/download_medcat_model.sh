#!/usr/bin/env bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
# ─────────────────────────────────────────────────────────────────────────────
# scripts/download_medcat_model.sh
#
# Downloads a MedCAT model pack via the KCL licence portal and places it in
# artifacts/medcat_models/ ready for docker compose.
#
# Auth flow:
#   1. GET  /auth-callback-api?key=LICENCE_KEY  → extracts csrftoken
#   2. POST /download-model (form payload)       → streams the model file
#
# All config is read from ../.env at the project root.
# The model choice can be overridden with a CLI argument.
#
# Usage (from anywhere inside medical-experiments/):
#   ./scripts/download_medcat_model.sh
#   ./scripts/download_medcat_model.sh snomed_uk
#
# Available models:
#   umls_small   UMLS subset — disorders, symptoms, medications (~1 GB)
#   umls_full    Full UMLS, 4M+ concepts (~5 GB)
#   snomed_int   SNOMED International, trained on MIMIC-III (~2 GB)
#   snomed_uk    SNOMED UK v2 2025, UMLS 2024AA, trained on MIMIC-IV (~2 GB)
#   snomed_mimic SNOMED International v2 2025, trained on MIMIC-IV (~2 GB)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve directories ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ENV="${PROJECT_DIR}/.env"
DEST_DIR="${PROJECT_DIR}/artifacts/medcat_models"

# ── KCL portal endpoints ──────────────────────────────────────────────────────
KCL_AUTH_URL="https://medcat.sites.er.kcl.ac.uk/auth-callback-api"
KCL_DOWNLOAD_URL="https://medcat.sites.er.kcl.ac.uk/download-model"
KCL_DOMAIN="medcat.sites.er.kcl.ac.uk"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}  →${RESET} $*"; }
success() { echo -e "${GREEN}  ✓${RESET} $*"; }
die()     { echo -e "${RED}  ✗ ERROR:${RESET} $*" >&2; exit 1; }

# ── Load project .env ─────────────────────────────────────────────────────────
if [[ ! -f "${PROJECT_ENV}" ]]; then
  die ".env not found at ${PROJECT_ENV}\n\n  Create it from the example:\n    cp ${PROJECT_DIR}/.env.example ${PROJECT_ENV}\n  Then fill in your credentials."
fi

set -a
# shellcheck source=/dev/null
source "${PROJECT_ENV}"
set +a

# ── Validate required credentials ─────────────────────────────────────────────
[[ -z "${UMLS_API_KEY:-}"  ]] && die "UMLS_API_KEY is not set in .env\n  Register at: https://uts.nlm.nih.gov/uts/profile"
[[ -z "${MEDCAT_FIRST_NAME:-}"   ]] && die "MEDCAT_FIRST_NAME is not set in .env"
[[ -z "${MEDCAT_LAST_NAME:-}"    ]] && die "MEDCAT_LAST_NAME is not set in .env"
[[ -z "${MEDCAT_EMAIL:-}"        ]] && die "MEDCAT_EMAIL is not set in .env"

# ── CLI argument overrides MEDCAT_MODEL from .env ─────────────────────────────
MODEL="${1:-${MEDCAT_MODEL:-snomed_mimic}}"

# ── Model registry ────────────────────────────────────────────────────────────
# Values are the modelpack identifiers sent in the KCL POST form.
# Update here when CogStack releases new versions.
declare -A MODEL_PACKS=(
  [umls_small]="MedCAT_UMLS_AllTypes_2020AB"
  [umls_full]="MedCAT_UMLS_AllTypes_2020AB_full"
  [snomed_int]="v2-SNOMED-2025-MIMIC"
  [snomed_uk]="v2-SNOMED-UK-2025-MIMIC"
  [snomed_mimic]="v2-SNOMED-2025-MIMIC"
)

declare -A MODEL_FILENAMES=(
  [umls_small]="umls_small_2020AB.zip"
  [umls_full]="umls_full_2020AB.zip"
  [snomed_int]="snomed_int_2025.zip"
  [snomed_uk]="snomed_uk_2025.zip"
  [snomed_mimic]="snomed_mimic_2025.zip"
)

# ── Validate model choice ─────────────────────────────────────────────────────
if [[ -z "${MODEL_PACKS[$MODEL]+_}" ]]; then
  echo -e "${RED}  ✗ ERROR:${RESET} Unknown model '${MODEL}'"
  echo ""
  echo "  Available models:"
  for key in umls_small umls_full snomed_int snomed_uk snomed_mimic; do
    printf "    %-14s  %s\n" "$key" "${MODEL_PACKS[$key]}"
  done
  echo ""
  echo "  Set MEDCAT_MODEL in .env or pass as a CLI argument."
  exit 1
fi

MODELPACK="${MODEL_PACKS[$MODEL]}"
FILENAME="${MODEL_FILENAMES[$MODEL]}"
DEST_PATH="${DEST_DIR}/${FILENAME}"
CONTAINER_PATH="/cat/models/${FILENAME}"

# ── Print resolved config ─────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────"
echo "  MedCAT model downloader"
echo "────────────────────────────────────────────────────────────"
echo "  Config from : ${PROJECT_ENV}"
echo "  Model       : ${MODEL}  (${MODELPACK})"
echo "  File        : ${FILENAME}"
echo "  Destination : ${DEST_PATH}"
echo "────────────────────────────────────────────────────────────"
echo ""

mkdir -p "${DEST_DIR}"

# ── Skip if already downloaded ────────────────────────────────────────────────
if [[ -f "${DEST_PATH}" ]]; then
  success "Model already present — skipping download."
  echo "    Delete the file and re-run to force a fresh download:"
  echo "      rm ${DEST_PATH}"
else

  # ── Temp files (cleaned up on exit) ────────────────────────────────────────
  COOKIE_JAR=$(mktemp /tmp/medcat_cookies.XXXXXX)
  AUTH_HEADERS=$(mktemp /tmp/medcat_auth_hdrs.XXXXXX)
  AUTH_BODY=$(mktemp /tmp/medcat_auth_body.XXXXXX)
  DL_HEADERS=$(mktemp /tmp/medcat_dl_hdrs.XXXXXX)
  trap 'rm -f "$COOKIE_JAR" "$AUTH_HEADERS" "$AUTH_BODY" "$DL_HEADERS"' EXIT

  # ── Step 1: GET auth-callback-api — obtain csrftoken ─────────────────────
  info "Authenticating with KCL licence portal (${UMLS_API_KEY:0:6}...)..."

  AUTH_HTTP=$(curl \
    --silent \
    --location \
    --write-out "%{http_code}" \
    --cookie-jar  "${COOKIE_JAR}" \
    --dump-header "${AUTH_HEADERS}" \
    --output      "${AUTH_BODY}" \
    --header "Accept: application/json, text/html, */*" \
    --header "User-Agent: MedCAT-Downloader/1.0" \
    --header "Referer: https://${KCL_DOMAIN}/" \
    --get \
    --data-urlencode "key=${UMLS_API_KEY}" \
    "${KCL_AUTH_URL}")

  if [[ "${AUTH_HTTP}" != "200" && "${AUTH_HTTP}" != "302" ]]; then
    echo "  Auth response body:"; cat "${AUTH_BODY}"
    die "Authentication failed (HTTP ${AUTH_HTTP}). Check your UMLS_API_KEY."
  fi

  # ── Extract CSRF token (4 fallback strategies) ────────────────────────────
  CSRF_TOKEN=""

  # 1. curl cookie jar (netscape format — token is the last whitespace-separated field)
  CSRF_TOKEN=$(awk '/csrftoken/{print $NF}' "${COOKIE_JAR}" 2>/dev/null | head -1 || true)

  # 2. Set-Cookie response header
  if [[ -z "${CSRF_TOKEN}" ]]; then
    CSRF_TOKEN=$(grep -i 'set-cookie:.*csrftoken' "${AUTH_HEADERS}" \
      | sed -E 's/.*csrftoken=([^;,[:space:]]+).*/\1/' \
      | head -1 || true)
  fi

  # 3. JSON body  { "csrftoken": "..." } / { "token": "..." }
  if [[ -z "${CSRF_TOKEN}" ]]; then
    CSRF_TOKEN=$(grep -oE '"(csrftoken|csrf_token|token)"\s*:\s*"[A-Za-z0-9_-]{20,}"' "${AUTH_BODY}" \
      | grep -oE '[A-Za-z0-9_-]{20,}' \
      | head -1 || true)
  fi

  # 4. Raw regex on body
  if [[ -z "${CSRF_TOKEN}" ]]; then
    CSRF_TOKEN=$(grep -oE 'csrftoken["\s:=]+[A-Za-z0-9_-]{20,}' "${AUTH_BODY}" \
      | grep -oE '[A-Za-z0-9_-]{20,}' \
      | head -1 || true)
  fi

  [[ -z "${CSRF_TOKEN}" ]] && {
    echo "  Auth headers:"; cat "${AUTH_HEADERS}"
    echo "  Auth body:";    cat "${AUTH_BODY}"
    die "Could not extract csrftoken from auth response."
  }

  success "Authenticated. Token: ${CSRF_TOKEN:0:12}..."

  # ── Step 2: POST download-model — stream model file ───────────────────────
  info "Requesting model pack '${MODELPACK}'..."

  DL_HTTP=$(curl \
    --silent \
    --location \
    --progress-bar \
    --write-out "%{http_code}" \
    --cookie      "${COOKIE_JAR}" \
    --cookie      "csrftoken=${CSRF_TOKEN}" \
    --dump-header "${DL_HEADERS}" \
    --output      "${DEST_PATH}" \
    --header "User-Agent: MedCAT-Downloader/1.0" \
    --header "Referer: https://${KCL_DOMAIN}/" \
    --header "X-CSRFToken: ${CSRF_TOKEN}" \
    --header "Accept: application/octet-stream, application/zip, */*" \
    --data-urlencode "csrfmiddlewaretoken=${CSRF_TOKEN}" \
    --data-urlencode "first_name=${MEDCAT_FIRST_NAME}" \
    --data-urlencode "last_name=${MEDCAT_LAST_NAME}" \
    --data-urlencode "email=${MEDCAT_EMAIL}" \
    --data-urlencode "affiliation=${MEDCAT_AFFILIATION:-}" \
    --data-urlencode "funder=${MEDCAT_FUNDER:-}" \
    --data-urlencode "use_case=${MEDCAT_USE_CASE:-}" \
    --data-urlencode "modelpack=${MODELPACK}" \
    --data-urlencode "consent=on" \
    "${KCL_DOWNLOAD_URL}")

  echo ""

  if [[ "${DL_HTTP}" == "403" ]]; then
    rm -f "${DEST_PATH}"
    die "403 Forbidden — licence key rejected or token expired."
  fi
  if [[ "${DL_HTTP}" != "200" ]]; then
    rm -f "${DEST_PATH}"
    die "Download request failed (HTTP ${DL_HTTP})."
  fi

  # Guard against the server returning an HTML error page instead of a file
  CONTENT_TYPE=$(grep -i '^content-type:' "${DL_HEADERS}" | tail -1 \
    | sed 's/content-type://I;s/[[:space:]]//g')
  if echo "${CONTENT_TYPE}" | grep -qi 'text/html\|text/plain'; then
    echo "  Server returned (first 20 lines):"; head -20 "${DEST_PATH}"
    rm -f "${DEST_PATH}"
    die "Expected a binary file but got '${CONTENT_TYPE}'.\n  Check the modelpack identifier or your registered email."
  fi

  # Rename to Content-Disposition filename if the server provides one
  CD_FILE=$(grep -i 'content-disposition:' "${DL_HEADERS}" \
    | grep -oiE 'filename\*?=([^;[:space:]]+)' \
    | sed -E "s/filename\*?=//I;s/^UTF-8''//I;s/[\"']//g" \
    | head -1 || true)
  if [[ -n "${CD_FILE}" && "${CD_FILE}" != "${FILENAME}" ]]; then
    mv "${DEST_PATH}" "${DEST_DIR}/${CD_FILE}"
    DEST_PATH="${DEST_DIR}/${CD_FILE}"
    CONTAINER_PATH="/cat/models/${CD_FILE}"
    info "Renamed to '${CD_FILE}' (from Content-Disposition header)"
  fi

  success "Download complete."
fi

# ── Zip integrity check ────────────────────────────────────────────────────────
if command -v unzip &>/dev/null; then
  info "Verifying zip integrity..."
  if ! unzip -t "${DEST_PATH}" &>/dev/null; then
    die "Zip integrity check failed.\n  The file may be corrupt or the server returned an error page.\n  Delete ${DEST_PATH} and re-run."
  fi
  success "Zip OK."
fi

# ── Write MEDCAT_MODEL_PACK_PATH back into .env ───────────────────────────────
KEY="MEDCAT_MODEL_PACK_PATH"
if grep -qE "^${KEY}=" "${PROJECT_ENV}"; then
  sed -i.bak "s|^${KEY}=.*|${KEY}=${CONTAINER_PATH}|" "${PROJECT_ENV}"
  rm -f "${PROJECT_ENV}.bak"
  info "Updated  ${KEY} in .env"
else
  printf "\n%s=%s\n" "${KEY}" "${CONTAINER_PATH}" >> "${PROJECT_ENV}"
  info "Appended ${KEY} to .env"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────"
echo "  Ready. To start the stack:"
echo ""
echo "    docker compose up -d"
echo ""
echo "  Verify medcat-service loaded the model (allow 30–90s):"
echo "    curl http://localhost:8004/api/info"
echo "────────────────────────────────────────────────────────────"
echo ""
