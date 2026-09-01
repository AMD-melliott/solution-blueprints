// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

/**
 * Client-side note sufficiency check.
 *
 * IMPORTANT — this is a FRONTEND-ONLY heuristic. The orchestrator's
 * AnalyzeResponse (schemas.py) has no sufficiency field; the backend does not
 * return this verdict. The Figma "Insufficient data" gate (screenshot 5) is a
 * UI affordance we compute here from the note text.
 *
 * Design intent (from the screenshot): a note is "insufficient" when it lacks
 * the objective sections a diagnostic workup needs — Vital Signs, Physical
 * Examination, and Labs/ECG. The 28yo chest-pain sample has none of these and
 * is the case shown blocked.
 *
 * CAVEAT: section detection from free text is inherently brittle. A note that
 * contains vitals without an explicit heading, or uses an unanticipated label,
 * can be wrongly flagged. We therefore detect both explicit headings AND
 * content signatures (e.g. "BP 120/80", "HR 92") to reduce false negatives.
 * Whether a flagged note should HARD-block Analyze or merely warn is a product
 * decision — see SUFFICIENCY_BLOCKS below to flip the behavior in one place.
 */

export interface SufficiencyResult {
  sufficient: boolean;
  missing: string[]; // human-readable labels for the chips, e.g. "Vital Signs"
}

/**
 * If true, an insufficient note disables the Analyze button (matches the
 * current Figma screenshot). If false, the banner still shows but Analyze
 * stays enabled (advisory mode). One-line switch for product to revisit.
 */
export const SUFFICIENCY_BLOCKS = true;

interface SectionCheck {
  label: string;
  // any heading pattern OR any content pattern present => section is present
  headings: RegExp[];
  content: RegExp[];
}

const REQUIRED_SECTIONS: SectionCheck[] = [
  {
    label: "Vital Signs",
    headings: [/\bvital\s*signs?\b/i, /\bvitals\b/i, /\bV\/S\b/],
    content: [
      /\bBP\b[\s:]*\d{2,3}\s*\/\s*\d{2,3}/i, // BP 118/76
      /\bblood\s+pressure\b[\s:]*\d{2,3}\s*\/\s*\d{2,3}/i,
      /\b(HR|heart\s+rate)\b[\s:]*\d{2,3}/i,
      /\b(RR|respiratory\s+rate)\b[\s:]*\d{1,2}/i,
      /\b(temp(erature)?|T)\b[\s:]*\d{2,3}(\.\d)?\s*°?\s*[CF]/i,
      /\bSpO2?\b[\s:]*\d{2,3}\s*%/i,
    ],
  },
  {
    label: "Physical Examination",
    headings: [/\bphysical\s+exam(ination)?\b/i, /\bexam(ination)?\b\s*:/i, /\bP\/?E\b\s*:/, /\bon\s+examination\b/i],
    content: [/\bauscultation\b/i, /\bpalpation\b/i, /\bpercussion\b/i, /\bbreath\s+sounds\b/i],
  },
  {
    label: "Labs / ECG",
    headings: [
      /\blab(oratory)?\s*(results?|findings?|data)?\b/i,
      /\bECG\b/,
      /\bEKG\b/,
      /\bimaging\b/i,
      /\binvestigations?\b/i,
    ],
    content: [
      /\bWBC\b[\s:]*\d/i,
      /\bCRP\b[\s:]*\d/i,
      /\bESR\b[\s:]*\d/i,
      /\btroponin\b/i,
      /\bX-?ray\b/i,
      /\bST\s+elevation\b/i,
    ],
  },
];

const MIN_USEFUL_LENGTH = 120; // chars; below this a note is too thin to analyze

function sectionPresent(text: string, s: SectionCheck): boolean {
  return s.headings.some((re) => re.test(text)) || s.content.some((re) => re.test(text));
}

export function checkSufficiency(text: string): SufficiencyResult {
  const trimmed = text.trim();

  // An empty or trivially short note is insufficient regardless of sections.
  if (trimmed.length < MIN_USEFUL_LENGTH) {
    return {
      sufficient: false,
      missing: REQUIRED_SECTIONS.map((s) => s.label),
    };
  }

  const missing = REQUIRED_SECTIONS.filter((s) => !sectionPresent(trimmed, s)).map((s) => s.label);

  return { sufficient: missing.length === 0, missing };
}
