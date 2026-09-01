// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import styles from "./App.module.css";
import { Header } from "./components/Header/Header";
import { NotesSidebar } from "./components/NotesSidebar/NotesSidebar";
import { NoteViewer } from "./components/NoteViewer/NoteViewer";
import { PipelineOutput } from "./components/PipelineOutput/PipelineOutput";
import { AddNoteModal } from "./components/AddNoteModal/AddNoteModal";
import { useSystemHealth } from "./hooks/useSystemHealth";
import { useAnalyze } from "./hooks/useAnalyze";
import { SAMPLE_NOTES } from "./utils/sampleNotes";
import type { ClinicalNote } from "./utils/sampleNotes";
import { checkSufficiency } from "./utils/sufficiency";

export function App() {
  const health = useSystemHealth();
  const { state: analyze, run, showCached } = useAnalyze();

  const [notes, setNotes] = useState<ClinicalNote[]>(SAMPLE_NOTES);
  const [activeId, setActiveId] = useState<number | null>(SAMPLE_NOTES[0]?.id ?? null);
  const [analyzedIds, setAnalyzedIds] = useState<Set<number>>(new Set());
  const [modalOpen, setModalOpen] = useState(false);

  const activeNote = notes.find((n) => n.id === activeId) ?? null;

  useEffect(() => {
    if (activeId != null) showCached(activeId);
  }, [activeId, showCached]);

  const activeInsufficient = useMemo(
    () => (activeNote ? !checkSufficiency(activeNote.text).sufficient : false),
    [activeNote],
  );

  async function handleAnalyze() {
    if (!activeNote) return;
    await run(activeNote.id, activeNote.text);
    setAnalyzedIds((prev) => new Set(prev).add(activeNote.id));
  }

  function handleAddNote(title: string, text: string) {
    const id = Math.max(0, ...notes.map((n) => n.id)) + 1;
    const note: ClinicalNote = { id, title, meta: "New case", text };
    setNotes((prev) => [...prev, note]);
    setActiveId(id);
  }

  return (
    <div className={styles.app}>
      <Header health={health} />

      <div className={styles.layout}>
        <div className={styles.column}>
          <NotesSidebar
            notes={notes}
            activeId={activeId}
            analyzedIds={analyzedIds}
            onSelect={setActiveId}
            onNew={() => setModalOpen(true)}
          />
        </div>

        <div className={styles.column}>
          <NoteViewer note={activeNote} analyze={analyze} onAnalyze={handleAnalyze} />
        </div>

        <div className={styles.column}>
          <PipelineOutput analyze={analyze} insufficient={activeInsufficient} onRetry={handleAnalyze} />
        </div>
      </div>

      <AddNoteModal open={modalOpen} onClose={() => setModalOpen(false)} onAdd={handleAddNote} />
    </div>
  );
}
