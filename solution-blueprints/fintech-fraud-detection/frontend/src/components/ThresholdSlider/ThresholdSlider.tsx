// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import styles from "./ThresholdSlider.module.css";
import type { Thresholds } from "../../types/api";
import { InfoTip } from "../InfoTip/InfoTip";

const TIP_THRESHOLDS = (
  <>
    Two cutoffs on the 0–1 score that split transactions into Approve / Review / Block. Drag them — every metric
    recomputes live.
  </>
);
const TIP_REVIEW = (
  <>
    Scores at or above this go to the <strong>review queue</strong> (human check).
  </>
);
const TIP_BLOCK = (
  <>
    Scores at or above this are <strong>auto-blocked</strong>.
  </>
);
const TIP_RESET = <>Restore the model-recommended thresholds.</>;

const MIN_GAP = 0.02;
const DEBOUNCE_MS = 200;
const RECOMMENDED = { review: 0.5, fraud: 0.8 };

interface ThresholdSliderProps {
  review: number;
  fraud: number;
  onChange: (t: Thresholds) => void;
}

export function ThresholdSlider({ review, fraud, onChange }: ThresholdSliderProps) {
  const [localReview, setLocalReview] = useState(review);
  const [localFraud, setLocalFraud] = useState(fraud);
  const trackRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef<"review" | "fraud" | null>(null);
  const debounceRef = useRef<number | null>(null);

  useEffect(() => {
    if (draggingRef.current === null) {
      setLocalReview(review);
      setLocalFraud(fraud);
    }
  }, [review, fraud]);

  const scheduleServerUpdate = useCallback(
    (r: number, f: number) => {
      if (debounceRef.current !== null) window.clearTimeout(debounceRef.current);
      debounceRef.current = window.setTimeout(() => {
        onChange({ review: r, fraud: f });
        debounceRef.current = null;
      }, DEBOUNCE_MS);
    },
    [onChange],
  );

  const pctFromClientX = useCallback((clientX: number): number | null => {
    const track = trackRef.current;
    if (!track) return null;
    const rect = track.getBoundingClientRect();
    const pct = (clientX - rect.left) / rect.width;
    return Math.max(0, Math.min(1, pct));
  }, []);

  const applyPosition = useCallback(
    (pct: number, which: "review" | "fraud") => {
      let r = localReview;
      let f = localFraud;
      if (which === "review") {
        r = Math.max(0, Math.min(pct, f - MIN_GAP));
      } else {
        f = Math.min(1, Math.max(pct, r + MIN_GAP));
      }
      setLocalReview(r);
      setLocalFraud(f);
      scheduleServerUpdate(r, f);
    },
    [localReview, localFraud, scheduleServerUpdate],
  );

  useEffect(() => {
    function onMove(e: PointerEvent) {
      if (!draggingRef.current) return;
      const pct = pctFromClientX(e.clientX);
      if (pct !== null) applyPosition(pct, draggingRef.current);
    }
    function onUp() {
      draggingRef.current = null;
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [pctFromClientX, applyPosition]);

  const onTrackPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      const target = e.target as HTMLElement;
      if (target.dataset.thumb) return;
      const pct = pctFromClientX(e.clientX);
      if (pct === null) return;
      const distReview = Math.abs(pct - localReview);
      const distFraud = Math.abs(pct - localFraud);
      const which = distReview <= distFraud ? "review" : "fraud";
      applyPosition(pct, which);
      draggingRef.current = which;
    },
    [pctFromClientX, localReview, localFraud, applyPosition],
  );

  const rPct = localReview * 100;
  const fPct = localFraud * 100;

  const resetRecommended = useCallback(() => {
    setLocalReview(RECOMMENDED.review);
    setLocalFraud(RECOMMENDED.fraud);
    scheduleServerUpdate(RECOMMENDED.review, RECOMMENDED.fraud);
  }, [scheduleServerUpdate]);

  return (
    <div className={styles.wrap}>
      <div className={styles.head}>
        <div className={styles.title}>
          <span className={styles.titleLabel}>
            <InfoTip label='Score thresholds' title='Score thresholds' body={TIP_THRESHOLDS} />
          </span>
        </div>
        <div className={styles.values}>
          Review <span className={styles.vR}>≥ {localReview.toFixed(2)}</span>
          <span className={styles.sep} />
          Fraud <span className={styles.vF}>≥ {localFraud.toFixed(2)}</span>
        </div>
        <span className={styles.presetRow}>
          <button type='button' className={styles.preset} onClick={resetRecommended}>
            Reset to recommended
          </button>
          <InfoTip label='Recommended thresholds' title='Recommended thresholds' body={TIP_RESET} iconOnly />
        </span>
      </div>

      <div className={styles.sliderCol}>
        <div className={styles.zoneLabels}>
          <span className={styles.lgClean} style={{ left: "0%" }}>
            Approve
          </span>
          <span className={styles.lgReview} style={{ left: `${rPct}%` }}>
            <InfoTip label='Review' title='Review threshold' body={TIP_REVIEW} />
          </span>
          <span className={styles.lgFraud} style={{ left: `${fPct}%` }}>
            <InfoTip label='Block' title='Block threshold' body={TIP_BLOCK} />
          </span>
        </div>

        <div className={styles.slider} ref={trackRef} onPointerDown={onTrackPointerDown}>
          <div className={styles.track}>
            <div className={`${styles.seg} ${styles.segClean}`} style={{ width: `${rPct}%` }} />
            <div
              className={`${styles.seg} ${styles.segReview}`}
              style={{ left: `${rPct}%`, width: `${Math.max(0, fPct - rPct)}%` }}
            />
            <div
              className={`${styles.seg} ${styles.segFraud}`}
              style={{ left: `${fPct}%`, width: `${Math.max(0, 100 - fPct)}%` }}
            />
          </div>
          <button
            type='button'
            data-thumb='review'
            className={`${styles.thumb} ${styles.thumbReview}`}
            style={{ left: `${rPct}%` }}
            onPointerDown={(e) => {
              e.stopPropagation();
              (e.target as HTMLElement).setPointerCapture(e.pointerId);
              draggingRef.current = "review";
            }}
            aria-label='Review threshold'
          />
          <button
            type='button'
            data-thumb='fraud'
            className={`${styles.thumb} ${styles.thumbFraud}`}
            style={{ left: `${fPct}%` }}
            onPointerDown={(e) => {
              e.stopPropagation();
              (e.target as HTMLElement).setPointerCapture(e.pointerId);
              draggingRef.current = "fraud";
            }}
            aria-label='Fraud threshold'
          />
        </div>

        <div className={styles.ends}>
          <span>0</span>
          <span>1.00</span>
        </div>
      </div>
    </div>
  );
}
