// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import styles from "./AddNoteModal.module.css";

interface AddNoteModalProps {
  open: boolean;
  onClose: () => void;
  onAdd: (title: string, text: string) => void;
}

export function AddNoteModal({ open, onClose, onAdd }: AddNoteModalProps) {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");

  // reset fields whenever the modal is (re)opened
  useEffect(() => {
    if (open) {
      setTitle("");
      setText("");
    }
  }, [open]);

  // close on Escape
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const canAdd = text.trim().length > 0;

  function handleAdd() {
    if (!canAdd) return;
    onAdd(title.trim() || "Untitled note", text.trim());
    onClose();
  }

  function handleSchedule() {
    // No backend for scheduling — this is a UI-only stub per the design.
    console.log("Schedule clicked", { title: title.trim(), text: text.trim() });
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <div className={styles.head}>
          <div className={styles.headText}>
            <span className={styles.title}>Add clinical note</span>
            <span className={styles.subtitle}>Create a new case for pipeline analysis.</span>
          </div>
          <button className={styles.close} onClick={onClose} aria-label='Close'>
            ✕
          </button>
        </div>

        <div className={styles.body}>
          <div className={styles.field}>
            <label className={styles.label}>Case title</label>
            <input
              className={styles.input}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder='e.g. 34yo Male — Rash + Joint Pain'
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>
              Clinical note{" "}
              <svg width='16' height='16' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'>
                <path
                  d='M6.06001 6.00016C6.21675 5.55461 6.52611 5.1789 6.93331 4.93958C7.34051 4.70027 7.81927 4.61279 8.28479 4.69264C8.75031 4.77249 9.17255 5.01451 9.47673 5.37585C9.7809 5.73718 9.94738 6.19451 9.94668 6.66683C9.94668 8.00016 7.94668 8.66683 7.94668 8.66683M8.00001 11.3335H8.00668M14.6667 8.00016C14.6667 11.6821 11.6819 14.6668 8.00001 14.6668C4.31811 14.6668 1.33334 11.6821 1.33334 8.00016C1.33334 4.31826 4.31811 1.3335 8.00001 1.3335C11.6819 1.3335 14.6667 4.31826 14.6667 8.00016Z'
                  stroke='#61656C'
                  strokeWidth='1.33333'
                  strokeLinecap='round'
                  strokeLinejoin='round'
                />
              </svg>
            </label>
            <textarea
              className={styles.textarea}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder='Paste or type the clinical note here…'
            />
            <span className={styles.helper}>SOAP format recommended. Free text supported.</span>
          </div>
        </div>

        <div className={styles.footer}>
          <button className={`${styles.btn} ${styles.btnSecondary}`} onClick={handleSchedule}>
            Schedule
          </button>
          <button className={`${styles.btn} ${styles.btnPrimary}`} onClick={handleAdd} disabled={!canAdd}>
            Add note
          </button>
        </div>
      </div>
    </div>
  );
}
