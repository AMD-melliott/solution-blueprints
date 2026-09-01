// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { checkSufficiency } from "./sufficiency";

export type NoteStatus = "ready" | "analyzed" | "insufficient";

/**
 * Derives the status badge for a note.
 * - analyzed: a pipeline result is cached for it
 * - insufficient: the client-side sufficiency check fails (no backend signal)
 * - ready: otherwise, eligible to analyze
 *
 * `analyzed` takes precedence: once a note has results we show those regardless
 * of the sufficiency heuristic.
 */
export function deriveNoteStatus(text: string, hasResult: boolean): NoteStatus {
  if (hasResult) return "analyzed";
  if (!checkSufficiency(text).sufficient) return "insufficient";
  return "ready";
}

export const STATUS_LABEL: Record<NoteStatus, string> = {
  ready: "Ready",
  analyzed: "Analyzed",
  insufficient: "Insufficient",
};
