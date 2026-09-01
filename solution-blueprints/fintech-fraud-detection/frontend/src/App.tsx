// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useState, useCallback } from "react";
import { Header } from "./components/Header/Header";
import { WarmupView } from "./components/WarmupView/WarmupView";
import { LiveView } from "./components/LiveView/LiveView";
import { useStreamEvents } from "./hooks/useStreamEvents";
import { useStreamControl } from "./hooks/useStreamControl";
import { useWarmupStatus } from "./hooks/useWarmupStatus";
import type { TabId } from "./types/api";
import styles from "./App.module.css";

export default function App() {
  const [tab, setTab] = useState<TabId>("warmup");
  const stream = useStreamEvents();
  const control = useStreamControl();
  const warmup = useWarmupStatus();

  // Resuming from pause keeps the accumulated view; any other start (after a
  // stop, which rewinds the backend to the beginning) is a fresh run, so flush
  // the on-screen transactions and metrics before restarting.
  const handleStart = useCallback(() => {
    if (stream.streamState !== "paused") {
      stream.reset();
    }
    control.start();
  }, [stream.streamState, stream.reset, control.start]);

  return (
    <div className={styles.app} data-theme='dark'>
      <Header
        streamState={stream.streamState}
        connected={stream.connected}
        systemReady={warmup.ready}
        activeTab={tab}
        onChangeTab={setTab}
        onStart={handleStart}
        onPause={control.pause}
        onStop={control.stop}
      />
      <div className={styles.viewport}>
        {tab === "warmup" && (
          <WarmupView onGoLive={() => setTab("live")} status={warmup.status} unreachable={warmup.unreachable} />
        )}
        {tab === "live" && <LiveView stream={stream} />}
      </div>
    </div>
  );
}
