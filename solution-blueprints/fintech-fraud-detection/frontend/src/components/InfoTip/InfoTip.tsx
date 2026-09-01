// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { ReactNode } from "react";
import styles from "./InfoTip.module.css";
import { useTooltipController } from "./TooltipProvider";

const HOVER_OPEN_DELAY = 500;
const HOVER_CLOSE_DELAY = 200;
const GAP = 9;
const MARGIN = 8;

interface InfoTipProps {
  title: string;
  body: ReactNode;
  label: string;
  iconOnly?: boolean;
}

type Placement = "top" | "bottom";

interface Coords {
  left: number;
  top: number;
  placement: Placement;
  caretLeft: number;
}

export function InfoTip({ title, body, label, iconOnly = false }: InfoTipProps) {
  const id = useId();
  const { openId, openMode, open, close, pin, registerPopup } = useTooltipController();
  const isOpen = openId === id;
  const isPinned = isOpen && openMode === "pinned";

  const anchorRef = useRef<HTMLButtonElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const openTimer = useRef<number | null>(null);
  const closeTimer = useRef<number | null>(null);

  const [coords, setCoords] = useState<Coords | null>(null);

  const clearTimers = useCallback(() => {
    if (openTimer.current !== null) window.clearTimeout(openTimer.current);
    if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
    openTimer.current = null;
    closeTimer.current = null;
  }, []);

  useEffect(() => {
    registerPopup(id, isOpen ? popupRef.current : null);
    return () => registerPopup(id, null);
  }, [id, isOpen, registerPopup, coords]);

  useEffect(() => () => clearTimers(), [clearTimers]);

  const scheduleOpen = useCallback(() => {
    if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
    if (isOpen || openTimer.current !== null) return;
    openTimer.current = window.setTimeout(() => {
      open(id, "hover");
      openTimer.current = null;
    }, HOVER_OPEN_DELAY);
  }, [id, isOpen, open]);

  const scheduleClose = useCallback(() => {
    if (openTimer.current !== null) {
      window.clearTimeout(openTimer.current);
      openTimer.current = null;
    }
    if (isPinned) return;
    if (closeTimer.current !== null) return;
    closeTimer.current = window.setTimeout(() => {
      close(id);
      closeTimer.current = null;
    }, HOVER_CLOSE_DELAY);
  }, [id, isPinned, close]);

  const togglePin = useCallback(() => {
    clearTimers();
    if (isPinned) close(id);
    else pin(id);
  }, [clearTimers, isPinned, close, pin, id]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        togglePin();
      } else if (e.key === "Escape") {
        clearTimers();
        close(id);
      }
    },
    [togglePin, clearTimers, close, id],
  );

  useLayoutEffect(() => {
    if (!isOpen) {
      setCoords(null);
      return;
    }
    const anchor = anchorRef.current;
    const popup = popupRef.current;
    if (!anchor || !popup) return;

    const a = anchor.getBoundingClientRect();
    const pw = popup.offsetWidth;
    const ph = popup.offsetHeight;
    const anchorCx = a.left + a.width / 2;

    const roomAbove = a.top;
    const roomBelow = window.innerHeight - a.bottom;
    const placement: Placement = roomAbove < ph + GAP + MARGIN && roomBelow > roomAbove ? "bottom" : "top";
    const top = placement === "top" ? a.top - GAP - ph : a.bottom + GAP;

    let left = anchorCx - pw / 2;
    left = Math.max(MARGIN, Math.min(left, window.innerWidth - MARGIN - pw));
    const caretLeft = Math.max(12, Math.min(anchorCx - left, pw - 12));

    setCoords({ left, top, placement, caretLeft });
  }, [isOpen, title, body]);

  useEffect(() => {
    if (!isOpen) return;
    const reflow = () => {
      const anchor = anchorRef.current;
      const popup = popupRef.current;
      if (!anchor || !popup) return;
      const a = anchor.getBoundingClientRect();
      const pw = popup.offsetWidth;
      const ph = popup.offsetHeight;
      const anchorCx = a.left + a.width / 2;
      const roomAbove = a.top;
      const roomBelow = window.innerHeight - a.bottom;
      const placement: Placement = roomAbove < ph + GAP + MARGIN && roomBelow > roomAbove ? "bottom" : "top";
      const top = placement === "top" ? a.top - GAP - ph : a.bottom + GAP;
      let left = anchorCx - pw / 2;
      left = Math.max(MARGIN, Math.min(left, window.innerWidth - MARGIN - pw));
      const caretLeft = Math.max(12, Math.min(anchorCx - left, pw - 12));
      setCoords({ left, top, placement, caretLeft });
    };
    window.addEventListener("scroll", reflow, true);
    window.addEventListener("resize", reflow);
    return () => {
      window.removeEventListener("scroll", reflow, true);
      window.removeEventListener("resize", reflow);
    };
  }, [isOpen]);

  return (
    <span
      className={styles.wrap}
      data-tip-anchor={id}
      onMouseEnter={scheduleOpen}
      onMouseLeave={scheduleClose}
      onClick={togglePin}
    >
      {!iconOnly && <span className={styles.labelText}>{label}</span>}
      <button
        ref={anchorRef}
        type='button'
        className={styles.icon}
        aria-label={`About: ${label}`}
        aria-expanded={isOpen}
        aria-describedby={isOpen ? `${id}-desc` : undefined}
        onFocus={() => open(id, "hover")}
        onBlur={() => !isPinned && close(id)}
        onKeyDown={onKeyDown}
      >
        <svg width='14' height='14' viewBox='0 0 16 16' fill='none' aria-hidden='true'>
          <circle cx='8' cy='8' r='6.4' stroke='currentColor' strokeWidth='1.3' />
          <circle cx='8' cy='5.1' r='0.95' fill='currentColor' />
          <path d='M8 7.2v4.0' stroke='currentColor' strokeWidth='1.3' strokeLinecap='round' />
        </svg>
      </button>

      {isOpen &&
        createPortal(
          <div
            ref={popupRef}
            id={`${id}-desc`}
            role='tooltip'
            className={`${styles.popup} ${coords?.placement === "bottom" ? styles.placeBottom : styles.placeTop}`}
            style={
              coords
                ? { left: coords.left, top: coords.top, ["--caret-left" as string]: `${coords.caretLeft}px` }
                : { left: -9999, top: -9999 }
            }
            onMouseEnter={scheduleOpen}
            onMouseLeave={scheduleClose}
          >
            {isPinned && (
              <button type='button' className={styles.close} aria-label='Close' onClick={() => close(id)}>
                ✕
              </button>
            )}
            <div className={styles.popTitle}>{title}</div>
            <div className={styles.popBody}>{body}</div>
          </div>,
          document.body,
        )}
    </span>
  );
}
