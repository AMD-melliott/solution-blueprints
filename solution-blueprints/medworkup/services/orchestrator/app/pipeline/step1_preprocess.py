# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Step 1 – Advanced Preprocessing

Two-path design:
  Fast path  → scispaCy deterministic NLP (always runs)
  Enhanced path → MedGemma abbreviation/normalisation (opt-in via flag)

Public interface is unchanged: preprocess(text, ...) → PreprocessedText
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple

from app.schemas import NegationSpan, PreprocessedText, SectionMap

logger = logging.getLogger(__name__)

# ── scispaCy lazy singleton ───────────────────────────────────────────────────
# Loaded once on first call; avoids import-time cost when unused.
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy

            # en_core_sci_sm is the lightweight scispaCy model.
            # Falls back to en_core_web_sm if the sci model is absent.
            try:
                _nlp = spacy.load("en_core_sci_sm")
                logger.info("[Preprocess] Loaded scispaCy model: en_core_sci_sm")
            except OSError:
                _nlp = spacy.load("en_core_web_sm")
                logger.warning(
                    "[Preprocess] en_core_sci_sm not found; using en_core_web_sm. "
                    "Install scispaCy for better biomedical NLP."
                )
        except ImportError:
            logger.warning("[Preprocess] spacy not installed; using regex fallback.")
            _nlp = None
    return _nlp


# ── Static abbreviation table ─────────────────────────────────────────────────
# Extended from original; add new entries here – no other code changes needed.
ABBREVIATION_MAP: Dict[str, str] = {
    # Symptoms / signs
    r"\bSOB\b": "shortness of breath",
    r"\bCP\b": "chest pain",
    r"\bN/V\b": "nausea and vomiting",
    r"\bN&V\b": "nausea and vomiting",
    r"\bHA\b": "headache",
    r"\bAMS\b": "altered mental status",
    r"\bLOC\b": "loss of consciousness",
    r"\bDOE\b": "dyspnea on exertion",
    r"\bPND\b": "paroxysmal nocturnal dyspnea",
    r"\bLE\b": "lower extremity",
    r"\bBLE\b": "bilateral lower extremities",
    # History
    r"\bHx\b": "history",
    r"\bPMHx\b": "past medical history",
    r"\bPMH\b": "past medical history",
    r"\bFHx\b": "family history",
    r"\bFH\b": "family history",
    r"\bSHx\b": "social history",
    r"\bSH\b": "social history",
    r"\bHPI\b": "history of present illness",
    r"\bROS\b": "review of systems",
    # Conditions
    r"\bHTN\b": "hypertension",
    r"\bDM\b": "diabetes mellitus",
    r"\bDM2\b": "type 2 diabetes mellitus",
    r"\bT2DM\b": "type 2 diabetes mellitus",
    r"\bCAD\b": "coronary artery disease",
    r"\bCHF\b": "congestive heart failure",
    r"\bHFrEF\b": "heart failure with reduced ejection fraction",
    r"\bHFpEF\b": "heart failure with preserved ejection fraction",
    r"\bCOPD\b": "chronic obstructive pulmonary disease",
    r"\bPE\b": "pulmonary embolism",
    r"\bDVT\b": "deep vein thrombosis",
    r"\bMI\b": "myocardial infarction",
    r"\bSTEMI\b": "ST-elevation myocardial infarction",
    r"\bNSTEMI\b": "non-ST-elevation myocardial infarction",
    r"\bCVA\b": "cerebrovascular accident",
    r"\bTIA\b": "transient ischemic attack",
    r"\bURTI\b": "upper respiratory tract infection",
    r"\bUTI\b": "urinary tract infection",
    r"\bAKI\b": "acute kidney injury",
    r"\bCKD\b": "chronic kidney disease",
    r"\bAFib\b": "atrial fibrillation",
    r"\bAF\b": "atrial fibrillation",
    r"\bVT\b": "ventricular tachycardia",
    r"\bVF\b": "ventricular fibrillation",
    r"\bGERD\b": "gastroesophageal reflux disease",
    r"\bIBD\b": "inflammatory bowel disease",
    r"\bRA\b": "rheumatoid arthritis",
    r"\bSLE\b": "systemic lupus erythematosus",
    r"\bMS\b": "multiple sclerosis",
    # Vitals / labs
    r"\bBP\b": "blood pressure",
    r"\bHR\b": "heart rate",
    r"\bRR\b": "respiratory rate",
    r"\bTemp\b": "temperature",
    r"\bO2\b": "oxygen",
    r"\bSpO2\b": "oxygen saturation",
    r"\bSaO2\b": "oxygen saturation",
    r"\bWBC\b": "white blood cell count",
    r"\bRBC\b": "red blood cell count",
    r"\bHgb\b": "hemoglobin",
    r"\bHb\b": "hemoglobin",
    r"\bHct\b": "hematocrit",
    r"\bPlt\b": "platelet count",
    r"\bINR\b": "international normalized ratio",
    r"\bPT\b": "prothrombin time",
    r"\bPTT\b": "partial thromboplastin time",
    r"\bBUN\b": "blood urea nitrogen",
    r"\bCr\b": "creatinine",
    r"\bNa\b": "sodium",
    r"\bK\b": "potassium",
    r"\bCl\b": "chloride",
    r"\bHCO3\b": "bicarbonate",
    r"\bLFTs\b": "liver function tests",
    r"\bTSH\b": "thyroid stimulating hormone",
    r"\bHbA1c\b": "hemoglobin A1c",
    r"\bBNP\b": "B-type natriuretic peptide",
    r"\bTroponin\b": "troponin",
    r"\bCRP\b": "C-reactive protein",
    r"\bESR\b": "erythrocyte sedimentation rate",
    # Imaging / procedures
    r"\bCXR\b": "chest X-ray",
    r"\bECG\b": "electrocardiogram",
    r"\bEKG\b": "electrocardiogram",
    r"\bEcho\b": "echocardiogram",
    r"\bCT\b": "computed tomography",
    r"\bCTA\b": "computed tomography angiography",
    r"\bMRI\b": "magnetic resonance imaging",
    r"\bMRA\b": "magnetic resonance angiography",
    r"\bUS\b": "ultrasound",
    r"\bTTE\b": "transthoracic echocardiogram",
    r"\bTEE\b": "transesophageal echocardiogram",
    r"\bV/Q\b": "ventilation-perfusion scan",
    r"\bPCI\b": "percutaneous coronary intervention",
    r"\bCABG\b": "coronary artery bypass grafting",
    r"\bCPR\b": "cardiopulmonary resuscitation",
    r"\bICU\b": "intensive care unit",
    r"\bED\b": "emergency department",
    r"\bOR\b": "operating room",
    # Medications / routes
    r"\bPO\b": "by mouth",
    r"\bIV\b": "intravenous",
    r"\bIM\b": "intramuscular",
    r"\bSQ\b": "subcutaneous",
    r"\bSC\b": "subcutaneous",
    r"\bPRN\b": "as needed",
    r"\bQD\b": "once daily",
    r"\bBID\b": "twice daily",
    r"\bTID\b": "three times daily",
    r"\bQID\b": "four times daily",
    r"\bNPO\b": "nothing by mouth",
    r"\bRx\b": "prescription",
    r"\bDx\b": "diagnosis",
    r"\bPx\b": "prognosis",
    r"\bTx\b": "treatment",
    r"\bMg\b": "milligrams",
    r"\bmcg\b": "micrograms",
}

# SOAP section patterns
_SECTION_PATTERNS: Dict[str, re.Pattern] = {
    "subjective": re.compile(r"^(subjective|cc|chief complaint|hpi|s\s*:)", re.IGNORECASE | re.MULTILINE),
    "objective": re.compile(r"^(objective|physical exam|pe|vitals|o\s*:)", re.IGNORECASE | re.MULTILINE),
    "assessment": re.compile(r"^(assessment|impression|a\s*:)", re.IGNORECASE | re.MULTILINE),
    "plan": re.compile(r"^(plan|management|p\s*:)", re.IGNORECASE | re.MULTILINE),
}

# Negation detection
_NEGATION_TRIGGERS: List[str] = [
    r"no\b",
    r"not\b",
    r"denies?\b",
    r"without\b",
    r"absent\b",
    r"negative\s+for\b",
    r"rules?\s+out\b",
    r"ruled\s+out\b",
    r"never\b",
    r"none\b",
    r"neither\b",
    r"nor\b",
    r"free\s+of\b",
    r"unremarkable\s+for\b",
    r"no\s+evidence\s+of\b",
]
_NEGATION_RE = re.compile(
    r"(" + "|".join(_NEGATION_TRIGGERS) + r")\s+(.{3,80}?)(?=[,\.;:\n]|$)",
    re.IGNORECASE,
)

# Garbled / de-identification artefacts to strip
_NOISE_RE = re.compile(
    r"\[(?:REDACTED|PHI|NAME|DATE|MRN)[^\]]{0,100}\]" r"|\bXXX+\b" r"|\b\d{10,}\b",  # long numeric strings (MRNs etc.)
    re.IGNORECASE,
)


# ── Text cleaning ─────────────────────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Remove noise, normalise whitespace, standardise punctuation."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = _NOISE_RE.sub(" ", text)
    # Collapse runs of dashes/underscores used as dividers
    text = re.sub(r"[-_]{3,}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Abbreviation expansion ────────────────────────────────────────────────────


def _expand_abbreviations(text: str) -> Tuple[str, Dict[str, str]]:
    applied: Dict[str, str] = {}
    for pattern, expansion in ABBREVIATION_MAP.items():
        new_text, n = re.subn(pattern, expansion, text)
        if n > 0:
            abbrev = re.sub(r"\\b", "", pattern)
            applied[abbrev] = expansion
            text = new_text
    return text, applied


# ── Section detection ─────────────────────────────────────────────────────────


def _detect_sections(text: str) -> SectionMap:
    positions: Dict[str, int] = {}
    for section, pat in _SECTION_PATTERNS.items():
        m = pat.search(text)
        if m:
            positions[section] = m.start()

    if not positions:
        return SectionMap(unclassified=text)

    ordered = sorted(positions.items(), key=lambda x: x[1])
    sections: Dict[str, str] = {}
    for i, (name, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        body = re.sub(_SECTION_PATTERNS[name], "", text[start:end], count=1).strip()
        sections[name] = body

    return SectionMap(
        subjective=sections.get("subjective"),
        objective=sections.get("objective"),
        assessment=sections.get("assessment"),
        plan=sections.get("plan"),
        unclassified=None,
    )


# ── Sentence segmentation ─────────────────────────────────────────────────────


def _segment_sentences_spacy(text: str) -> List[str]:
    nlp = _get_nlp()
    if nlp is None:
        return _segment_sentences_regex(text)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _segment_sentences_regex(text: str) -> List[str]:
    """Regex fallback when spaCy is unavailable."""
    sentences = re.split(r"(?<=[.!?])\s{1,20}(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


# ── Negation detection ────────────────────────────────────────────────────────


def _detect_negations(text: str) -> List[NegationSpan]:
    spans: List[NegationSpan] = []
    for m in _NEGATION_RE.finditer(text):
        spans.append(
            NegationSpan(
                text=m.group(2).strip(),
                start=m.start(2),
                end=m.end(2),
                trigger=m.group(1),
            )
        )
    return spans


# ── MedGemma enhanced normalisation (optional) ────────────────────────────────


async def _medgemma_enhance(
    text: str,
    llm_client,
) -> Tuple[str, Dict[str, str]]:
    """
    Ask MedGemma to expand unknown abbreviations and standardise terminology.
    Returns (enhanced_text, {abbrev: expansion}).
    Only called when use_medgemma=True and llm_client is provided.
    """
    prompt = (
        "You are a clinical NLP assistant. Given the following clinical note, "
        "expand any abbreviations not in the standard list and standardise "
        "non-standard medical terminology. "
        "Return ONLY a JSON object with two keys:\n"
        '  - "text": the full note with expansions inline\n'
        '  - "expansions": {"ABBREV": "expansion"} for each change made\n'
        "Do not add commentary. Clinical note:\n\n"
        f"{text}"
    )
    try:
        raw = await llm_client.chat(prompt, max_tokens=len(text) * 2 + 256)
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        enhanced = data.get("text", text)
        expansions = data.get("expansions", {})
        return enhanced, expansions
    except Exception as exc:
        logger.warning("[Preprocess/MedGemma] Enhancement failed (%s); using static expansion.", exc)
        return text, {}


# ── scispaCy abbreviation detection (abbrev component) ───────────────────────


def _detect_abbreviations_spacy(text: str) -> Dict[str, str]:
    """
    Use scispaCy abbreviation detector (AbbreviationDetector) when available.
    Returns {short_form: long_form}.
    """
    nlp = _get_nlp()
    if nlp is None:
        return {}
    # AbbreviationDetector is an optional scispaCy component
    if "abbreviation_detector" not in nlp.pipe_names:
        try:
            from scispacy.abbreviation import AbbreviationDetector  # noqa: F401

            nlp.add_pipe("abbreviation_detector")
            logger.info("[Preprocess] AbbreviationDetector added to pipeline")
        except (ImportError, Exception):
            return {}
    try:
        doc = nlp(text)
        return {str(abrv): str(abrv._.long_form) for abrv in doc._.abbreviations if str(abrv) not in ABBREVIATION_MAP}
    except Exception as exc:
        logger.debug("[Preprocess] AbbreviationDetector error: %s", exc)
        return {}


# ── public entry point ────────────────────────────────────────────────────────


async def preprocess(
    text: str,
    expand_abbreviations: bool = True,
    use_medgemma: bool = False,
    llm_client=None,
) -> PreprocessedText:
    """
    Main preprocessing entry point.

    Args:
        text: Raw clinical text.
        expand_abbreviations: Apply static abbreviation table (always fast).
        use_medgemma: Call MedGemma for intelligent abbreviation expansion.
                      Only active when llm_client is provided.
        llm_client: LLMClient instance (required when use_medgemma=True).

    Returns:
        PreprocessedText with all fields populated.
    """
    original = text
    normalized = _clean_text(text)

    # ── Fast path: static expansion ──────────────────────────────────────────
    applied_abbrevs: Dict[str, str] = {}
    if expand_abbreviations:
        normalized, applied_abbrevs = _expand_abbreviations(normalized)

    # Detect additional abbreviations with scispaCy
    spacy_abbrevs = _detect_abbreviations_spacy(normalized)
    # Apply scispaCy-detected abbreviations (they are already long-form in text;
    # just record them for traceability)
    applied_abbrevs.update(spacy_abbrevs)

    # ── Enhanced path: MedGemma (optional) ───────────────────────────────────
    if use_medgemma and llm_client is not None:
        normalized, llm_abbrevs = await _medgemma_enhance(normalized, llm_client)
        applied_abbrevs.update(llm_abbrevs)
        logger.info("[Preprocess/MedGemma] %d additional expansions", len(llm_abbrevs))

    # ── scispaCy sentence segmentation ───────────────────────────────────────
    sentences = _segment_sentences_spacy(normalized)

    # ── Downstream NLP ────────────────────────────────────────────────────────
    section_map = _detect_sections(normalized)
    negations = _detect_negations(normalized)

    logger.info(
        "[Preprocess] sentences=%d negations=%d abbrevs=%d",
        len(sentences),
        len(negations),
        len(applied_abbrevs),
    )

    return PreprocessedText(
        original=original,
        normalized=normalized,
        sentences=sentences,
        section_map=section_map,
        negations=negations,
        abbreviations_expanded=applied_abbrevs,
    )
