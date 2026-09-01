// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { band, type Band } from "./computeMetrics";

export type DecisionLabel = "Approved" | "Review" | "Blocked";
export type ResultLabel = "Caught" | "Missed" | "False alert" | "Reviewed clean" | "Clean" | null;

export function decisionFromBand(b: Band): DecisionLabel {
  if (b === "BLOCK") return "Blocked";
  if (b === "REVIEW") return "Review";
  return "Approved";
}

export function resultFromBandGt(b: Band, isFraud: number | null): ResultLabel {
  if (isFraud === null) return null;
  if (isFraud === 1) {
    if (b === "BLOCK") return "Caught";
    if (b === "REVIEW") return "Caught";
    return "Missed";
  }
  // clean
  if (b === "BLOCK") return "False alert";
  if (b === "REVIEW") return "Reviewed clean";
  return "Clean";
}

export interface RowVerdict {
  band: Band;
  decision: DecisionLabel;
  result: ResultLabel;
}

export function classifyRow(score: number, isFraud: number | null, reviewTh: number, fraudTh: number): RowVerdict {
  const b = band(score, reviewTh, fraudTh);
  return { band: b, decision: decisionFromBand(b), result: resultFromBandGt(b, isFraud) };
}

export function isFlagged(score: number, reviewTh: number, fraudTh: number): boolean {
  const b = band(score, reviewTh, fraudTh);
  return b === "REVIEW" || b === "BLOCK";
}
