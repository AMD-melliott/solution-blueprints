# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Deterministic rules engine for fraud detection.

Design (per 2026-06-23 baseline, revised 2026-06-26 — see RULES_ENGINE_CHANGES):
  * CONFIG-DRIVEN. Thresholds, enabled rules and the auto-block tier are loaded
    from rules_config.yaml (tuned by tune_rules.py on the IEEE-CIS holdout).
    RulesEngine() with no args reproduces the legacy all-defaults behaviour.
  * AUTO-BLOCK tier (2026-06-26 decision). A transaction's fraud_score = the
    calibrated historical fraud rate of the exact combination of stateless rules
    it fires (scorecard.json). It auto-blocks (action=BLOCK → DECLINE) when that
    score ≥ config block.fraud_threshold (θ). Shipping θ=0.25: ~72% block-set
    precision, FP/TP≈0.38 on the holdout — far inside the FP≤X·TP budget (X=5).
    With no block config the tier is empty and the engine is a pure overlay.
  * Below the block gate, rules remain an OVERLAY on the ML decision: REVIEW
    escalates APPROVE→REVIEW, MONITOR annotates only.
  * The data-derived rules (R030+) are tuned EMPIRICALLY on the IEEE-CIS train
    split (base fraud rate 3.5%); they are statistical signals, so only deep
    high-precision COMBINATIONS (not single rules) ever clear the block gate.

Evaluation tiers:
  HIGH   → BLOCK    : auto-decline (scorecard signature score ≥ θ)
  MEDIUM → REVIEW   : escalate (override ML APPROVE → REVIEW)
  LOW    → MONITOR  : log / annotate only

REVIEW tier (MEDIUM) — high-precision escalation queue (~22% precision / 27%
recall as a group on IEEE-CIS train). These read RAW columns and fire statelessly:
  R030 : ProductCD=C & C1>10 & risky recipient domain  (~65% precision, 18x lift)
  R031 : C4>2 & C7>1  (counting-feature combo)         (~42% precision, 12x lift)
  R032 : card6=credit & risky domain & high-risk hour  (~26% precision,  7x lift)
  R022 : new-account high activity (D1<7 & C1>5)        (~21% precision,  6x lift)

MONITOR tier (LOW) — annotate / log only (never escalates on its own):
  R033 : ProductCD=C  (broad catch-all)                (~12% precision, 39% recall)
  R034 : risky recipient email domain                  (gmail/outlook/icloud/hotmail)
  R013 : high-risk transaction hour 05-10 (data-derived; was 02-05)
  R005, R006, R010, R011, R012, R015 — velocity/amount rules DEMOTED from
        REVIEW (2026-06-23): <11% precision on IEEE-CIS, were the dominant
        false-positive source ("374 FP, zero real fraud" — FT Sync 2026-06-22).
  R024 : unusual channel (device shift + amount spike).

Disabled (kept for SOW completeness but NOT evaluated — 0-1% precision, noise):
  R020 (amount splitting), R021 (round amounts), R023 (frequent tx).

Rules NOT implementable from IEEE-CIS dataset (require external data):
  R001 (sanction list — needs IP/country blacklist)
  R002 (stolen card   — needs stolen-card DB)
  R003 (multiple failures — needs session failure counter)
  R004 (impossible travel — needs GPS/IP geolocation)
  R014 (unusual MCC   — no MCC column in dataset)
  R016 (VPN/Proxy/Tor — no IP address in dataset)
  R017 (profile change — no profile-event stream)
  R025 (geo vs IP     — no GPS or IP geolocation)
"""

import json
import os
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class RuleMatch:
    rule_id: str
    description: str
    risk_level: str  # HIGH | MEDIUM | LOW
    action: str  # BLOCK | REVIEW | MONITOR


# Recipient email domains with empirically elevated fraud rates on IEEE-CIS
# (R_emaildomain fraud rate vs 3.5% base: gmail 11.9%, outlook 16.5%,
#  icloud 12.9%, hotmail 7.8%). Used by R030 / R032 / R034.
RISKY_RECIPIENT_DOMAINS = frozenset({"gmail.com", "outlook.com", "icloud.com", "hotmail.com"})

# Rules whose combinations form the auto-block scorecard. All stateless (read raw
# columns of a single tx), so the signature is order-independent and reproducible.
# MUST match STATELESS_RULES in tune_rules.py (the harness that calibrates scores).
# R035 (M4=M2) deliberately EXCLUDED: it is fraud-enriched (3.3x) but ~redundant
# with ProductCD=C / risky-email at the high-precision end, so adding it to the
# block signature only fragments buckets without catching new fraud (verified
# 2026-06-26: block@0.25 796→789). It stays a MONITOR annotation instead.
STATELESS_RULES = frozenset({"R030", "R031", "R032", "R022", "R033", "R034"})


def _load_yaml_or_json(path: str) -> dict:
    """Load a config/scorecard file. JSON is valid YAML, so yaml.safe_load reads
    both; falls back to the json module when PyYAML is unavailable (stdlib sandbox).
    Returns {} if the file does not exist."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        text = fh.read()
    try:
        import yaml  # optional dependency; present in the backend image

        return yaml.safe_load(text) or {}
    except ImportError:
        return json.loads(text) if text.strip() else {}


def assert_fp_budget(tp: int, fp: int, x: float) -> None:
    """Raise AssertionError if the auto-block set violates FP <= X*TP.

    This is the configurable coefficient: FP no more than X times the fraud the
    engine catches. Equivalent to a block-set precision floor of 1/(1+X).
    Used by validate/tune as an invariant — NOT a per-transaction runtime gate."""
    if tp == 0:
        if fp > 0:
            raise AssertionError(f"FP budget violated: {fp} FP with 0 TP (X={x})")
        return
    if fp / tp > x:
        raise AssertionError(f"FP budget violated: FP/TP={fp / tp:.2f} > X={x}")


# ── Velocity tracker ─────────────────────────────────────────────────────────


class VelocityTracker:
    """
    Thread-safe per-card ring buffer of recent transaction signals.

    Each stored entry is a tuple:
      (ts: float, amount: float, r_emaildomain: str|None,
       addr2: float|None, device_type: str|None)

    ts is TransactionDT (seconds from dataset epoch) when available;
    falls back to wall-clock time.  Entries older than 24 h are evicted lazily.
    """

    _KEEP_SECONDS: float = 86_400.0  # 24 h retention window

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[int, deque] = defaultdict(lambda: deque())

    def push(
        self,
        card1: Optional[int],
        ts: float,
        amount: float,
        r_email: Optional[str],
        addr2: Optional[float],
        device_type: Optional[str],
    ) -> None:
        if card1 is None:
            return
        with self._lock:
            buf = self._store[card1]
            buf.append((ts, amount, r_email, addr2, device_type))
            cutoff = ts - self._KEEP_SECONDS
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def recent(self, card1: Optional[int], window_sec: float, current_ts: float) -> list[tuple]:
        """Return entries for *card1* within the last *window_sec* seconds."""
        if card1 is None:
            return []
        cutoff = current_ts - window_sec
        with self._lock:
            return [e for e in self._store.get(card1, deque()) if e[0] >= cutoff]

    def clear(self) -> None:
        """Drop all per-card history (used when the stream is stopped/restarted)."""
        with self._lock:
            self._store.clear()


# ── Rules engine ─────────────────────────────────────────────────────────────


class RulesEngine:
    """
    Evaluates deterministic fraud rules.

    Usage pattern (ML-primary, rules as overlay — 2026-06-23 decision):
        engine = RulesEngine(card_stats=bundle.get("card_stats", {}))

        matches = engine.evaluate(tx)       # check rules
        engine.record(tx)                   # update velocity state

        ml_decision = model.predict(tx)     # ML is the PRIMARY driver — always runs
        worst = RulesEngine.worst_action(matches)
        # Overlay: a REVIEW match escalates an ML "APPROVE" → "REVIEW";
        # a MONITOR match is logged only. Rules never DECLINE on their own.
    """

    # ── Thresholds (match the rule spec) ─────────────────────────────────────
    _LARGE_AMT: float = 10_000.0  # R010 fallback, R011
    _AMT_MULTIPLIER: float = 5.0  # R010: amount > card_mean * 5
    _RISK_HOUR_START: int = 5  # R013/R032: data-derived high-risk window (was 02-05)
    _RISK_HOUR_END: int = 10
    _R030_C1_MIN: float = 10.0  # R030: ProductCD=C counting-feature threshold
    _R031_C4_MIN: float = 2.0  # R031: counting-feature combo
    _R031_C7_MIN: float = 1.0
    _ROUND_MODULO: float = 1_000.0  # R021
    _ROUND_THRESHOLD: int = 3  # R021: ≥ 3 round txs in 24 h
    _NEW_ACCT_D1_DAYS: float = 7.0  # R022: D1 proxy for account age
    _NEW_ACCT_C1_COUNT: float = 5.0  # R022: C1 proxy for tx count
    _FREQ_TX_1H: int = 15  # R023
    _RECIPIENTS_24H: int = 10  # R015
    _SMALL_AMT: float = 1.00  # R006
    _CARD_TEST_MIN: int = 3  # R006: count in 5 min
    _DUP_WINDOW_SEC: float = 60.0  # R005
    _SPLIT_THRESHOLD: float = 10_000.0  # R020

    def __init__(
        self,
        card_stats: Optional[dict] = None,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        card_stats: mapping card1 (int) → {"mean": float, "std": float}.
        Populated from the model bundle's training-time per-card statistics.
        When absent, R010 falls back to a global large-amount threshold.

        config / config_path: tuning configuration (see rules_config.yaml). Every
        key is optional and OVERRIDES a hardcoded default, so RulesEngine() with no
        args reproduces the legacy behaviour. Resolution order:
          explicit `config` dict  >  `config_path`  >  rules_config.yaml next to
          this module  >  {} (all defaults, no auto-block tier).
        """
        self._card_stats: dict = card_stats or {}
        self._velocity = VelocityTracker()

        if config is None:
            path = config_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules_config.yaml")
            config = _load_yaml_or_json(path)
        self._config: dict = config or {}

        # ── Per-rule thresholds — override class defaults from config.thresholds ──
        th = self._config.get("thresholds") or {}
        self._R030_C1_MIN = float(th.get("R030", {}).get("c1_min", self._R030_C1_MIN))
        self._R031_C4_MIN = float(th.get("R031", {}).get("c4_min", self._R031_C4_MIN))
        self._R031_C7_MIN = float(th.get("R031", {}).get("c7_min", self._R031_C7_MIN))
        self._RISK_HOUR_START = int(th.get("R032", {}).get("hour_start", self._RISK_HOUR_START))
        self._RISK_HOUR_END = int(th.get("R032", {}).get("hour_end", self._RISK_HOUR_END))
        self._NEW_ACCT_D1_DAYS = float(th.get("R022", {}).get("d1_max", self._NEW_ACCT_D1_DAYS))
        self._NEW_ACCT_C1_COUNT = float(th.get("R022", {}).get("c1_min", self._NEW_ACCT_C1_COUNT))

        # ── Enabled rules — config.tiers.{review,monitor}; None = legacy default ──
        tiers = self._config.get("tiers") or {}
        if tiers:
            self._enabled: Optional[set] = set(tiers.get("review", [])) | set(tiers.get("monitor", []))
        else:
            self._enabled = None  # evaluate everything currently wired (legacy)

        # ── Auto-block tier — calibrated signature scorecard ─────────────────────
        block = self._config.get("block") or {}
        self._fraud_threshold: Optional[float] = block.get("fraud_threshold")
        self._scorecard: dict = {}
        if self._fraud_threshold is not None:
            sc_path = block.get("scorecard_path", "scorecard.json")
            if not os.path.isabs(sc_path):
                base = os.path.dirname(config_path) if config_path else os.path.dirname(os.path.abspath(__file__))
                sc_path = os.path.join(base, sc_path)
            self._scorecard = (_load_yaml_or_json(sc_path) or {}).get("scores", {})

    def _on(self, rule_id: str) -> bool:
        """True if *rule_id* is enabled (legacy default = all wired rules on)."""
        return self._enabled is None or rule_id in self._enabled

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, tx: dict) -> list[RuleMatch]:
        """
        Evaluate all rules against *tx* and return triggered RuleMatch list.
        Order: REVIEW (MEDIUM) → MONITOR (LOW). No BLOCK tier.
        Does NOT modify velocity state — call record() separately.

        IMPORTANT — these rules read RAW dataset columns (TransactionDT, C1, C4,
        C7, D1, R_emaildomain, card6, ProductCD). They must NOT be rewired to the
        engineered features that are mis-synced at inference (tx_hour, D1_norm,
        amt_zscore, product_risk_score, tx_day, tx_day_week — always 0 on the
        inference path); otherwise R013/R022/R032 would silently stop firing.
        """
        ts = self._ts(tx)
        card1 = _int_or_none(tx.get("card1"))
        amount = _float(tx.get("TransactionAmt"))

        out: list[RuleMatch] = []

        # ── REVIEW (MEDIUM) — high-precision escalation queue.
        #    Data-derived stateless rules (fire on a single tx) + R022.
        #    Tuned on IEEE-CIS train; ~22% precision / 27% recall as a group.
        if self._on("R030"):
            self._r030_prodc_count_email(tx, out)  # ~65% precision
        if self._on("R031"):
            self._r031_counting_combo(tx, out)  # ~42% precision
        if self._on("R032"):
            self._r032_credit_hour_email(tx, out)  # ~26% precision
        if self._on("R022"):
            self._r022_new_account_activity(tx, out)  # ~21% precision

        # ── MONITOR (LOW) — annotate / log only (never escalates on its own).
        if self._on("R033"):
            self._r033_product_c(tx, out)
        if self._on("R034"):
            self._r034_risky_recipient(tx, out)
        if self._on("R035"):
            self._r035_m4_match(tx, out)
        if self._on("R013"):
            self._r013_risk_hour(tx, out)
        # Demoted from REVIEW (2026-06-23 re-analysis): <11% precision on
        # IEEE-CIS — these were the main source of false-positive escalations
        # ("374 FP, zero real fraud" — FT Sync 2026-06-22). R006/R010/R011 are
        # disabled in the shipped config (near-zero precision) but stay wired.
        if self._on("R005"):
            self._r005_duplicate(ts, card1, amount, out)
        if self._on("R006"):
            self._r006_card_testing(ts, card1, amount, out)
        if self._on("R010"):
            self._r010_anomalous_amount(card1, amount, out)
        if self._on("R011"):
            self._r011_new_device_large_amount(tx, amount, out)
        if self._on("R012"):
            self._r012_country_change(tx, ts, card1, out)
        if self._on("R015"):
            self._r015_many_recipients(tx, ts, card1, out)
        if self._on("R024"):
            self._r024_unusual_channel(tx, ts, card1, out)

        # ── Disabled — 0–1% precision on IEEE-CIS (pure noise). Methods kept
        #    for SOW completeness but intentionally NOT evaluated.
        # self._r020_amount_splitting(ts, card1, amount, out)
        # self._r021_round_amounts(ts, card1, amount, out)
        # self._r023_frequent_transactions(ts, card1, out)

        # ── BLOCK (HIGH) — auto-block when the fired-rule combination's calibrated
        #    historical fraud rate clears the configured fraud_threshold (θ).
        self._apply_block_tier(out)

        return out

    def fraud_score(self, matches: list[RuleMatch]) -> Optional[float]:
        """Calibrated fraud probability for the combination of stateless rules in
        *matches* (from the scorecard), or None if no scorecard is loaded / no
        stateless rule fired. Exposed for UI explainability."""
        if not self._scorecard:
            return None
        fired = sorted(m.rule_id for m in matches if m.rule_id in STATELESS_RULES)
        if not fired:
            return None
        return self._scorecard.get("+".join(fired))

    def _apply_block_tier(self, out: list) -> None:
        """Append a BLOCK match if the fired stateless-rule signature scores >= θ."""
        if self._fraud_threshold is None or not self._scorecard:
            return
        score = self.fraud_score(out)
        if score is not None and score >= self._fraud_threshold:
            sig = "+".join(sorted(m.rule_id for m in out if m.rule_id in STATELESS_RULES))
            out.append(
                RuleMatch(
                    rule_id="R900",
                    description=(
                        f"Auto-block: rule combination [{sig}] has {score:.0%} historical "
                        f"fraud rate (≥ {self._fraud_threshold:.0%} threshold)"
                    ),
                    risk_level="HIGH",
                    action="BLOCK",
                )
            )

    def reset(self) -> None:
        """Clear all accumulated velocity/history state.

        Called when the demo stream is stopped/restarted so duplicate- and
        velocity-based rules (R005, R006, R012, R015, R020-R024) don't fire on a
        replay because of the previous run's transactions.
        """
        self._velocity.clear()

    def record(self, tx: dict) -> None:
        """Push *tx* into the velocity tracker.  Call once after evaluate()."""
        self._velocity.push(
            card1=_int_or_none(tx.get("card1")),
            ts=self._ts(tx),
            amount=_float(tx.get("TransactionAmt")),
            r_email=tx.get("R_emaildomain"),
            addr2=tx.get("addr2"),
            device_type=tx.get("DeviceType"),
        )

    @staticmethod
    def hard_blocked(matches: list[RuleMatch]) -> bool:
        """
        True if any match is a hard BLOCK. With an auto-block config loaded
        (block.fraud_threshold set), the R900 signature rule emits BLOCK when a
        transaction's calibrated fraud_score clears θ; the predictor then DECLINEs
        without waiting for ML. With no block config this stays False (pure overlay).
        """
        return any(m.action == "BLOCK" for m in matches)

    @staticmethod
    def worst_action(matches: list[RuleMatch]) -> Optional[str]:
        """Return the most severe action across all triggered matches, or None."""
        _rank = {"BLOCK": 3, "REVIEW": 2, "MONITOR": 1}
        if not matches:
            return None
        return max(matches, key=lambda m: _rank.get(m.action, 0)).action

    # ── Timestamp helper ──────────────────────────────────────────────────────

    @staticmethod
    def _ts(tx: dict) -> float:
        dt = tx.get("TransactionDT")
        return float(dt) if dt is not None else time.time()

    @staticmethod
    def _hour(tx: dict) -> Optional[int]:
        """Relative transaction hour 0-23 = (TransactionDT // 3600) % 24, or None."""
        dt = tx.get("TransactionDT")
        if dt is None:
            return None
        try:
            return int(float(dt) // 3600) % 24
        except (TypeError, ValueError):
            return None

    # ── Data-derived rules (IEEE-CIS-tuned, stateless) ───────────────────────

    def _r030_prodc_count_email(self, tx: dict, out: list) -> None:
        """
        R030: ProductCD=C AND C1>10 AND recipient domain in the risky set.
        Strongest empirical signal on train: ~65% precision, 18x lift,
        ~13% of all fraud caught at ~0.7% volume.
        """
        if tx.get("ProductCD") != "C":
            return
        c1 = _float(tx.get("C1"))
        if c1 <= self._R030_C1_MIN:
            return
        if (tx.get("R_emaildomain") or "") in RISKY_RECIPIENT_DOMAINS:
            out.append(
                RuleMatch(
                    rule_id="R030",
                    description=(
                        f"High-risk profile: ProductCD=C, C1={c1:.0f}>10, " f"recipient={tx.get('R_emaildomain')}"
                    ),
                    risk_level="MEDIUM",
                    action="REVIEW",
                )
            )

    def _r031_counting_combo(self, tx: dict, out: list) -> None:
        """
        R031: C4>2 AND C7>1 — counting-feature combo (~42% precision, 12x lift).
        C4/C7 separate fraud sharply (ok p99 ≈ 3/2 vs fraud p99 ≈ 206/91).
        """
        c4, c7 = tx.get("C4"), tx.get("C7")
        if c4 is None or c7 is None:
            return
        if _float(c4) > self._R031_C4_MIN and _float(c7) > self._R031_C7_MIN:
            out.append(
                RuleMatch(
                    rule_id="R031",
                    description=f"Counting-feature anomaly: C4={_float(c4):.0f}, C7={_float(c7):.0f}",
                    risk_level="MEDIUM",
                    action="REVIEW",
                )
            )

    def _r032_credit_hour_email(self, tx: dict, out: list) -> None:
        """
        R032: card6=credit AND high-risk hour (05-10) AND risky recipient domain.
        ~26% precision, 7.5x lift — the hour signal is useful as a combiner.
        """
        if tx.get("card6") != "credit":
            return
        h = self._hour(tx)
        if h is None or not (self._RISK_HOUR_START <= h <= self._RISK_HOUR_END):
            return
        if (tx.get("R_emaildomain") or "") in RISKY_RECIPIENT_DOMAINS:
            out.append(
                RuleMatch(
                    rule_id="R032",
                    description=f"Credit card at high-risk hour {h:02d}:xx to risky recipient",
                    risk_level="MEDIUM",
                    action="REVIEW",
                )
            )

    def _r033_product_c(self, tx: dict, out: list) -> None:
        """R033: ProductCD=C — broad catch-all (~12% precision, 39% recall, 3.3x lift)."""
        if tx.get("ProductCD") == "C":
            out.append(
                RuleMatch(
                    rule_id="R033",
                    description="ProductCD=C (elevated base fraud rate, ~3.3x)",
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    def _r034_risky_recipient(self, tx: dict, out: list) -> None:
        """R034: recipient email domain in the elevated-risk set."""
        dom = tx.get("R_emaildomain") or ""
        if dom in RISKY_RECIPIENT_DOMAINS:
            out.append(
                RuleMatch(
                    rule_id="R034",
                    description=f"Recipient domain {dom} (elevated fraud rate)",
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    def _r035_m4_match(self, tx: dict, out: list) -> None:
        """
        R035: M4 = 'M2'. M4 is a Vesta name/address 'match' flag (values M0/M1/M2);
        the M2 category is empirically fraud-enriched on IEEE-CIS (~11.4% fraud,
        3.3x base, ~60k rows). Weak alone (MONITOR) but a strong COMBINER — added
        to R022 it lifts precision 21%→61% (FP/TP 3.66→0.64). Data-derived,
        stateless. Inspired by the IEEE-CIS winners' use of M-flag signals; uses
        only the raw single-transaction column (no leakage).
        """
        if tx.get("M4") == "M2":
            out.append(
                RuleMatch(
                    rule_id="R035",
                    description="M4=M2 match-flag (elevated fraud rate ~3.3x)",
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    # ── MONITOR rules (velocity / behavioural, demoted from REVIEW) ──────────
    #    All of R005/R006/R010/R011/R012/R015 below were REVIEW until the
    #    2026-06-23 re-analysis. They score <11% precision on IEEE-CIS and were
    #    the dominant false-positive source, so they now only annotate (MONITOR)
    #    and never escalate the ML decision on their own.

    def _r005_duplicate(self, ts: float, card1: Optional[int], amount: float, out: list) -> None:
        """
        R005: Same card + same amount within 60 seconds.
        Full spec also requires same merchant; dataset has no merchant ID,
        so card1 + amount is used as a conservative proxy.
        BLOCK → REVIEW → MONITOR: card+amount alone is a weak proxy (~10%
        precision on IEEE-CIS); annotate only.
        """
        recent = self._velocity.recent(card1, self._DUP_WINDOW_SEC, ts)
        for entry_ts, entry_amt, *_ in recent:
            if abs(entry_amt - amount) < 0.01:
                out.append(
                    RuleMatch(
                        rule_id="R005",
                        description=(
                            f"Duplicate tx: card+amount (${amount:.2f}) " f"repeated within {int(ts - entry_ts)}s"
                        ),
                        risk_level="LOW",
                        action="MONITOR",
                    )
                )
                return

    def _r006_card_testing(self, ts: float, card1: Optional[int], amount: float, out: list) -> None:
        """
        R006: amount ≤ $1.00 AND ≥ 3 transactions in last 5 minutes.
        Micro-amounts probe whether a card is live before a large charge.
        BLOCK → REVIEW → MONITOR: velocity-only, near-zero precision on the
        replay stream; annotate only.
        """
        if amount > self._SMALL_AMT:
            return
        recent = self._velocity.recent(card1, 300.0, ts)
        if len(recent) >= self._CARD_TEST_MIN:
            out.append(
                RuleMatch(
                    rule_id="R006",
                    description=(f"Card testing: {len(recent)} tx in 5 min " f"at micro-amount ${amount:.2f}"),
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    def _r010_anomalous_amount(self, card1: Optional[int], amount: float, out: list) -> None:
        """
        R010: amount > card_mean * 5.
        card_stats loaded from training bundle; falls back to global $10k threshold.
        Demoted REVIEW → MONITOR: amount barely separates fraud in IEEE-CIS
        (fraud mean ≈ legit mean; fraud is smaller at the tail).
        """
        stats = self._card_stats.get(card1) if card1 is not None else None
        if stats:
            mean = stats.get("mean", 0.0)
            if mean > 0 and amount > mean * self._AMT_MULTIPLIER:
                out.append(
                    RuleMatch(
                        rule_id="R010",
                        description=(
                            f"Anomalous amount: ${amount:.2f} > " f"{self._AMT_MULTIPLIER}× card avg ${mean:.2f}"
                        ),
                        risk_level="LOW",
                        action="MONITOR",
                    )
                )
        elif amount > self._LARGE_AMT:
            out.append(
                RuleMatch(
                    rule_id="R010",
                    description=f"Large amount: ${amount:.2f} (no card history)",
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    def _r011_new_device_large_amount(self, tx: dict, amount: float, out: list) -> None:
        """
        R011: id_15 = "New" (identity-table device-seen flag) AND amount > $10k.
        id_15 values in dataset: "New" | "Found" | "NotFound".
        Demoted REVIEW → MONITOR: fires on ~0 rows (the >$10k gate is too rare).
        """
        if amount <= self._LARGE_AMT:
            return
        id_15 = tx.get("id_15") or tx.get("id15")
        if str(id_15).strip().lower() == "new":
            out.append(
                RuleMatch(
                    rule_id="R011",
                    description=f"New device (id_15=New) with large amount ${amount:.2f}",
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    def _r012_country_change(self, tx: dict, ts: float, card1: Optional[int], out: list) -> None:
        """
        R012: addr2 (billing country code) differs from the most recent
        addr2 seen for this card.  Detects sudden billing-country switches.
        """
        current = tx.get("addr2")
        if current is None or card1 is None:
            return
        recent = self._velocity.recent(card1, 86_400.0, ts)
        if not recent:
            return
        prev = recent[-1][3]  # addr2 is index 3 in the stored tuple
        if prev is not None and prev != current:
            out.append(
                RuleMatch(
                    rule_id="R012",
                    description=f"Country change: addr2 {prev} → {current}",
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    def _r013_risk_hour(self, tx: dict, out: list) -> None:
        """
        R013: Transaction hour in the data-derived high-risk window 05-10.
        IEEE-CIS fraud peaks at relative hours 05-10 (up to ~3x base rate),
        NOT the textbook 02-05 night window. Demoted to MONITOR — standalone
        precision is low (~7-10%); the hour is most useful as a combiner (R032).
        """
        h = self._hour(tx)
        if h is None:
            return
        if self._RISK_HOUR_START <= h <= self._RISK_HOUR_END:
            out.append(
                RuleMatch(
                    rule_id="R013",
                    description=f"High-risk hour: {h:02d}:xx (data-derived window 05-10)",
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    def _r015_many_recipients(self, tx: dict, ts: float, card1: Optional[int], out: list) -> None:
        """
        R015: > 10 distinct R_emaildomain values in last 24 h for this card.
        R_emaildomain is the best available recipient proxy in the dataset.
        """
        if card1 is None:
            return
        recent = self._velocity.recent(card1, 86_400.0, ts)
        domains = {e[2] for e in recent if e[2] is not None}
        if len(domains) > self._RECIPIENTS_24H:
            out.append(
                RuleMatch(
                    rule_id="R015",
                    description=f"Many recipients: {len(domains)} distinct R_emaildomain in 24 h",
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    # ── LOW rules ─────────────────────────────────────────────────────────────

    def _r020_amount_splitting(self, ts: float, card1: Optional[int], amount: float, out: list) -> None:
        """
        R020: Multiple small transactions summing above threshold within 1 h.
        Structuring pattern — keeps individual amounts below reporting limits.
        """
        if card1 is None:
            return
        recent = self._velocity.recent(card1, 3_600.0, ts)
        if len(recent) < 3:
            return
        total = sum(e[1] for e in recent) + amount
        max_single = max(max(e[1] for e in recent), amount)
        if total > self._SPLIT_THRESHOLD and max_single < self._SPLIT_THRESHOLD:
            out.append(
                RuleMatch(
                    rule_id="R020",
                    description=(
                        f"Amount splitting: {len(recent)+1} tx in 1 h, "
                        f"total=${total:.2f}, max single=${max_single:.2f}"
                    ),
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    def _r021_round_amounts(self, ts: float, card1: Optional[int], amount: float, out: list) -> None:
        """
        R021: amount % 1000 == 0 AND ≥ 3 such transactions in last 24 h.
        Round multiples of 1000 are a structuring / money-mule signal.
        """
        if amount % self._ROUND_MODULO != 0:
            return
        if card1 is None:
            out.append(
                RuleMatch(
                    rule_id="R021",
                    description=f"Round amount: ${amount:.0f}",
                    risk_level="LOW",
                    action="MONITOR",
                )
            )
            return
        recent = self._velocity.recent(card1, 86_400.0, ts)
        round_count = sum(1 for e in recent if e[1] % self._ROUND_MODULO == 0)
        # current tx not yet in buffer, so threshold is _ROUND_THRESHOLD - 1
        if round_count >= self._ROUND_THRESHOLD - 1:
            out.append(
                RuleMatch(
                    rule_id="R021",
                    description=(f"Round amounts: {round_count + 1} × $1k-multiple in 24 h, " f"latest=${amount:.0f}"),
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    def _r022_new_account_activity(self, tx: dict, out: list) -> None:
        """
        R022: account_age_days < 7 AND tx_count > 5.
        Proxy: D1 (days since last card use) < 7 AND C1 (cumulative count) > 5.
        D1=0 on first-ever use; C1 counts prior transactions on this card.
        Promoted MONITOR → REVIEW (2026-06-23): ~21% precision / ~20% recall on
        IEEE-CIS — a genuine stateless signal (reads raw D1/C1), strong enough
        to escalate. New-but-busy cards are a real fraud pattern here.
        """
        d1 = tx.get("D1")
        c1 = tx.get("C1")
        if d1 is None or c1 is None:
            return
        try:
            if float(d1) < self._NEW_ACCT_D1_DAYS and float(c1) > self._NEW_ACCT_C1_COUNT:
                out.append(
                    RuleMatch(
                        rule_id="R022",
                        description=(
                            f"New account high activity: D1={float(d1):.0f}d, " f"C1={float(c1):.0f} prior tx"
                        ),
                        risk_level="MEDIUM",
                        action="REVIEW",
                    )
                )
        except (TypeError, ValueError):
            # Non-numeric D1/C1 values mean this proxy cannot be evaluated for this transaction.
            return

    def _r023_frequent_transactions(self, ts: float, card1: Optional[int], out: list) -> None:
        """R023: > 15 transactions in the last 1 hour for this card."""
        if card1 is None:
            return
        recent = self._velocity.recent(card1, 3_600.0, ts)
        if len(recent) > self._FREQ_TX_1H:
            out.append(
                RuleMatch(
                    rule_id="R023",
                    description=f"Frequent transactions: {len(recent)} in last 1 h",
                    risk_level="LOW",
                    action="MONITOR",
                )
            )

    def _r024_unusual_channel(self, tx: dict, ts: float, card1: Optional[int], out: list) -> None:
        """
        R024: DeviceType differs from preferred channel AND amount > 2× recent avg.
        Proxy: DeviceType (mobile/desktop) as channel; preferred = most common in 24 h.
        """
        if card1 is None:
            return
        current_device = tx.get("DeviceType")
        if not current_device:
            return
        recent = self._velocity.recent(card1, 86_400.0, ts)
        if not recent:
            return
        device_counts = Counter(e[4] for e in recent if e[4] is not None)
        if not device_counts:
            return
        preferred = device_counts.most_common(1)[0][0]
        if preferred == current_device:
            return
        amounts = [e[1] for e in recent]
        avg = sum(amounts) / len(amounts)
        amount = _float(tx.get("TransactionAmt"))
        if avg > 0 and amount > avg * 2:
            out.append(
                RuleMatch(
                    rule_id="R024",
                    description=(
                        f"Unusual channel: {preferred} → {current_device}, " f"amount=${amount:.2f} > 2× avg ${avg:.2f}"
                    ),
                    risk_level="LOW",
                    action="MONITOR",
                )
            )


# ── Utilities ─────────────────────────────────────────────────────────────────


def _float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _int_or_none(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
