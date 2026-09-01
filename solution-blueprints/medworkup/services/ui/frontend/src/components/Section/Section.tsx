// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useState, type ReactNode } from "react";
import styles from "./Section.module.css";

type DotColor = "blue" | "green" | "amber" | "purple" | "red";

const DOT_COLOR: Record<DotColor, string> = {
  blue: "var(--blue-action)",
  green: "var(--green)",
  amber: "var(--amber)",
  purple: "var(--purple)",
  red: "var(--red)",
};

interface SectionProps {
  title: string;
  dot: DotColor;
  count?: number | null;
  defaultOpen?: boolean;
  children: ReactNode;
}

export function Section({ title, dot, count = null, defaultOpen = false, children }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={styles.section}>
      <div
        className={styles.head}
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen((o) => !o);
          }
        }}
      >
        <span aria-hidden style={{ width: 8, height: 8, borderRadius: "50%", background: DOT_COLOR[dot] }} />
        <span className={styles.title}>{title}</span>
        {count != null && <span className={styles.count}>{count}</span>}
        <span className={styles.chevron}>
          {open ? (
            <svg width='14' height='8' viewBox='0 0 14 8' fill='none' xmlns='http://www.w3.org/2000/svg'>
              <path d='M1 1L7 7L13 1' stroke='#61656C' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' />
            </svg>
          ) : (
            <svg width='8' height='14' viewBox='0 0 8 14' fill='none' xmlns='http://www.w3.org/2000/svg'>
              <path d='M1 13L7 7L1 1' stroke='#61656C' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' />
            </svg>
          )}
        </span>
      </div>
      {open && <div className={styles.body}>{children}</div>}
    </div>
  );
}
