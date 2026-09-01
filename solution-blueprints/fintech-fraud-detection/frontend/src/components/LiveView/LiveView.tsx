// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import styles from "./LiveView.module.css";
import { ThresholdSlider } from "../ThresholdSlider/ThresholdSlider";
import { MetricsStrip } from "../MetricsStrip/MetricsStrip";
import { TransactionTable } from "../TransactionTable/TransactionTable";
import { FraudDrawer } from "../FraudDrawer/FraudDrawer";
import type { StreamEventsState } from "../../hooks/useStreamEvents";
import type { ScoredTransaction, Thresholds } from "../../types/api";
import { computeMetrics } from "../../utils/computeMetrics";
import { isFlagged } from "../../utils/rowVerdict";
import { TooltipProvider } from "../InfoTip/TooltipProvider";

interface LiveViewProps {
  stream: StreamEventsState;
}

const DEFAULT_THRESHOLDS: Thresholds = { review: 0.5, fraud: 0.8 };

export function LiveView({ stream }: LiveViewProps) {
  const [selected, setSelected] = useState<ScoredTransaction | null>(null);

  const [thresholds, setThresholds] = useState<Thresholds>(DEFAULT_THRESHOLDS);

  const metrics = useMemo(
    () => computeMetrics(stream.metricRows, thresholds.review, thresholds.fraud, stream.metricsTruncated),
    [stream.metricRows, thresholds.review, thresholds.fraud, stream.metricsTruncated],
  );

  const fraudQueue = useMemo(
    () => stream.transactions.filter((t) => isFlagged(t.score, thresholds.review, thresholds.fraud)),
    [stream.transactions, thresholds.review, thresholds.fraud],
  );

  return (
    <TooltipProvider>
      <div className={styles.view}>
        <ThresholdSlider review={thresholds.review} fraud={thresholds.fraud} onChange={setThresholds} />

        <MetricsStrip m={metrics} />

        <div className={styles.cols}>
          <div className={styles.pane}>
            <div className={styles.paneHead}>
              <span className={styles.paneTitle}>Live Transaction Stream</span>
            </div>
            <div className={styles.tableWrap}>
              <TransactionTable
                rows={stream.transactions}
                variant='stream'
                reviewTh={thresholds.review}
                fraudTh={thresholds.fraud}
                onRowClick={setSelected}
                selectedId={selected?.transaction_id ?? null}
              />
            </div>
          </div>

          <div className={styles.pane}>
            <div className={styles.paneHead}>
              <span className={styles.paneTitle}>Review Queue</span>
              <span className={styles.pillCount}>
                <span className={styles.pillDot} />
                {fraudQueue.length} flagged
              </span>
            </div>
            <div className={styles.tableWrap}>
              <TransactionTable
                rows={fraudQueue}
                variant='queue'
                reviewTh={thresholds.review}
                fraudTh={thresholds.fraud}
                onRowClick={setSelected}
                selectedId={selected?.transaction_id ?? null}
              />
            </div>
          </div>
        </div>

        <FraudDrawer transaction={selected} fraudThreshold={thresholds.fraud} onClose={() => setSelected(null)} />
      </div>
    </TooltipProvider>
  );
}
