// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import styles from "./Header.module.css";
import { StreamControls } from "../StreamControls/StreamControls";
import type { StreamState, TabId } from "../../types/api";

interface HeaderProps {
  streamState: StreamState;
  connected: boolean;
  systemReady: boolean;
  activeTab: TabId;
  onChangeTab: (id: TabId) => void;
  onStart: () => void;
  onPause: () => void;
  onStop: () => void;
}

const TABS: { id: TabId; label: string }[] = [
  { id: "warmup", label: "Warm Up" },
  { id: "live", label: "Live" },
];

export function Header({
  streamState,
  connected,
  systemReady,
  activeTab,
  onChangeTab,
  onStart,
  onPause,
  onStop,
}: HeaderProps) {
  let badgeClass = styles.badgeReady;
  let badgeText = "System ready";
  if (!connected) {
    badgeClass = styles.badgeFail;
    badgeText = "Backend unreachable";
  } else if (!systemReady) {
    badgeClass = styles.badgePaused;
    badgeText = "Warming up";
  } else if (streamState === "running") {
    badgeClass = styles.badgeLive;
    badgeText = "Live";
  } else if (streamState === "paused") {
    badgeClass = styles.badgePaused;
    badgeText = "Paused";
  }

  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <div className={styles.logo}>
          FRAUD<em>LENS</em>
        </div>
        <div className={styles.tagline}>Real-time fraud scoring</div>
      </div>

      <nav className={styles.tabs}>
        {TABS.map((t) => (
          <button
            key={t.id}
            type='button'
            className={`${styles.tab} ${activeTab === t.id ? styles.tabActive : ""}`}
            onClick={() => onChangeTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className={styles.right}>
        {activeTab === "live" && (
          <StreamControls
            state={streamState}
            startDisabled={!systemReady}
            onStart={onStart}
            onPause={onPause}
            onStop={onStop}
          />
        )}
        <div className={`${styles.badge} ${badgeClass}`}>
          <span className={styles.dot} />
          <span>{badgeText}</span>
        </div>
      </div>
    </header>
  );
}
