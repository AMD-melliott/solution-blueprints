// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import styles from "./NotesSidebar.module.css";
import type { ClinicalNote } from "../../utils/sampleNotes";
import { deriveNoteStatus, STATUS_LABEL } from "../../utils/noteStatus";

interface NotesSidebarProps {
  notes: ClinicalNote[];
  activeId: number | null;
  analyzedIds: Set<number>;
  onSelect: (id: number) => void;
  onNew: () => void;
}

export function NotesSidebar({ notes, activeId, analyzedIds, onSelect, onNew }: NotesSidebarProps) {
  return (
    <>
      <div className={styles.head}>
        <span className={styles.title}>Clinical Notes</span>
        <button className={styles.newBtn} onClick={onNew}>
          <svg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'>
            <path
              d='M6.66668 0.833252V12.4999M0.833344 6.66659H12.5'
              stroke='#0086C9'
              strokeWidth='1.66667'
              strokeLinecap='round'
              strokeLinejoin='round'
            />
          </svg>
          New
        </button>
      </div>

      <div className={styles.list}>
        {notes.map((note) => {
          const status = deriveNoteStatus(note.text, analyzedIds.has(note.id));
          return (
            <div
              key={note.id}
              className={`${styles.item} ${note.id === activeId ? styles.active : ""}`}
              role="button"
              tabIndex={0}
              onClick={() => onSelect(note.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(note.id);
                }
              }}
            >
              <span className={`${styles.statusPill} ${styles[status]}`}>
                <span className={styles.statusDot} />
                {STATUS_LABEL[status]}
              </span>
              <div className={styles.itemTitle}>{note.title}</div>
              <div className={styles.itemMeta}>{note.meta}</div>
            </div>
          );
        })}
      </div>
    </>
  );
}
