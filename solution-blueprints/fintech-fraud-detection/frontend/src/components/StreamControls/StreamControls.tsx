// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import styles from "./StreamControls.module.css";
import type { StreamState } from "../../types/api";

interface StreamControlsProps {
  state: StreamState;
  /** Disables Start while the system is still warming up (Pause is unaffected) */
  startDisabled?: boolean;
  onStart: () => void;
  onPause: () => void;
  onStop: () => void;
}

const STATE_LABEL: Record<StreamState, string> = {
  idle: "Idle",
  running: "Running",
  paused: "Paused",
  stopped: "Stopped",
};

export function StreamControls({ state, startDisabled = false, onStart, onPause, onStop }: StreamControlsProps) {
  const isRunning = state === "running";
  const canStop = state === "running" || state === "paused";

  const labelClass = state === "running" ? styles.labelRunning : state === "paused" ? styles.labelPaused : "";

  return (
    <div className={styles.controls}>
      <span className={`${styles.label} ${labelClass}`}>{STATE_LABEL[state]}</span>
      <button
        type='button'
        className={`${styles.btn} ${isRunning ? styles.btnActive : ""}`}
        onClick={isRunning ? onPause : onStart}
        disabled={!isRunning && startDisabled}
        title={isRunning ? "Pause" : startDisabled ? "System is warming up" : "Start"}
        aria-label={isRunning ? "Pause stream" : "Start stream"}
      >
        {isRunning ? (
          <svg width='16' height='16' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'>
            <path
              fillRule='evenodd'
              clipRule='evenodd'
              d='M4 1C5.10457 1 6 1.89543 6 3V13C6 14.1046 5.10457 15 4 15C2.89543 15 2 14.1046 2 13V3C2 1.89543 2.89543 1 4 1ZM12 1C13.1046 1 14 1.89543 14 3V13C14 14.1046 13.1046 15 12 15C10.8954 15 10 14.1046 10 13V3C10 1.89543 10.8954 1 12 1Z'
              fill='#47CD89'
            />
          </svg>
        ) : (
          "▶"
        )}
      </button>
      <button
        type='button'
        className={`${styles.btn} ${styles.stopBtn}`}
        onClick={onStop}
        disabled={!canStop}
        title='Stop'
        aria-label='Stop stream'
      >
        <svg width='16' height='16' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'>
          <path
            d='M2 5.2C2 4.0799 2 3.51984 2.21799 3.09202C2.40973 2.71569 2.71569 2.40973 3.09202 2.21799C3.51984 2 4.0799 2 5.2 2H10.8C11.9201 2 12.4802 2 12.908 2.21799C13.2843 2.40973 13.5903 2.71569 13.782 3.09202C14 3.51984 14 4.0799 14 5.2V10.8C14 11.9201 14 12.4802 13.782 12.908C13.5903 13.2843 13.2843 13.5903 12.908 13.782C12.4802 14 11.9201 14 10.8 14H5.2C4.0799 14 3.51984 14 3.09202 13.782C2.71569 13.5903 2.40973 13.2843 2.21799 12.908C2 12.4802 2 11.9201 2 10.8V5.2Z'
            fill='white'
          />
        </svg>
      </button>
    </div>
  );
}
