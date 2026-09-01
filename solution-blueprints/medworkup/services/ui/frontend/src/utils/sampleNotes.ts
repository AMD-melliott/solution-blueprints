// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

/**
 * Preloaded sample clinical notes — transcribed verbatim from the legacy
 * services/ui/static/index.html SOAPS array. Kept preloaded so the app shows
 * real content on first load (and so the mock orchestrator can match on them).
 */

export interface ClinicalNote {
  id: number;
  title: string;
  meta: string; // category label shown under the title (Infectious / Cardiac / ...)
  text: string;
}

export const SAMPLE_NOTES: ClinicalNote[] = [
  {
    id: 1,
    title: "34yo Male — Rash + Arthralgia",
    meta: "Infectious",
    text: `PATIENT: Male, 34 years old. Forest ranger.

CHIEF COMPLAINT: Expanding skin rash on right thigh and bilateral knee joint pain for 10 days.

HISTORY OF PRESENT ILLNESS: Patient noticed a bull's-eye shaped rash on his right thigh approximately 8 days ago. The rash has been progressively expanding. He also reports bilateral knee joint pain and stiffness, worse in the morning. Associated fatigue and low-grade fever. Patient recalls a hiking trip in Connecticut 3 weeks ago with prolonged outdoor exposure in wooded areas. No tick bite was noticed, but exposure was significant.

PAST MEDICAL HISTORY: No significant past medical history. No prior joint disease.

MEDICATIONS: None currently.

ALLERGIES: Penicillin — skin rash.

SOCIAL HISTORY: Works as a forest ranger with frequent exposure to wooded environments. Non-smoker, occasional alcohol use.

VITAL SIGNS: Temperature 38.1°C, Blood Pressure 118/76 mmHg, Heart Rate 92 bpm, Respiratory Rate 16/min, SpO₂ 98% on room air.

PHYSICAL EXAMINATION: Erythema migrans — annular rash approximately 12 cm in diameter on right thigh with central clearing. Mild bilateral knee effusion noted on palpation. No neck stiffness. Lungs clear to auscultation.

LABORATORY RESULTS: WBC 11.2 × 10⁹/L (elevated), ESR 42 mm/hr (elevated), CRP 18 mg/L (elevated). Lyme ELISA sent — pending.

IMAGING: Bilateral knee X-ray: mild joint effusion bilaterally, no erosive changes, joint space preserved.`,
  },
  {
    id: 2,
    title: "67yo Female — Cough + Dyspnea",
    meta: "Respiratory",
    text: `PATIENT: Female, 67 years old. Retired school teacher.

CHIEF COMPLAINT: Productive cough and shortness of breath for 5 days.

HISTORY OF PRESENT ILLNESS: Patient presents with a 5-day history of productive cough with yellow-green sputum, progressive dyspnea on exertion, and pleuritic chest pain on the right side. Fever developed 3 days ago. She denies hemoptysis. No sick contacts reported.

PAST MEDICAL HISTORY: Type 2 Diabetes Mellitus (on metformin), Hypertension (on lisinopril). No prior pulmonary disease.

MEDICATIONS: Metformin 1000 mg BID, Lisinopril 10 mg QD.

VITAL SIGNS: Temperature 38.9°C, Blood Pressure 138/84 mmHg, Heart Rate 104 bpm, Respiratory Rate 22/min, SpO₂ 94% on room air.

PHYSICAL EXAMINATION: Decreased breath sounds and dullness to percussion at right lower lobe. Crackles heard on auscultation right base. No wheeze.

LABORATORY RESULTS: WBC 15.8 × 10⁹/L (elevated, neutrophilia), CRP 94 mg/L (markedly elevated), Procalcitonin 2.1 ng/mL (elevated). Blood cultures × 2 sent.

IMAGING: Chest X-ray: right lower lobe consolidation, no pleural effusion, no pneumothorax.`,
  },
  {
    id: 3,
    title: "52yo Male — Chest Pain + Diaphoresis",
    meta: "Cardiac",
    text: `PATIENT: Male, 52 years old. Accountant.

CHIEF COMPLAINT: Sudden-onset chest pain and diaphoresis for 45 minutes.

HISTORY OF PRESENT ILLNESS: Patient presents with sudden-onset crushing substernal chest pain radiating to the left arm and jaw, associated with diaphoresis and nausea. Onset at rest. No relief with position change. Denies shortness of breath. History of hypertension and hyperlipidemia.

MEDICATIONS: Amlodipine 5 mg QD, Atorvastatin 40 mg QD.

VITAL SIGNS: Blood Pressure 158/96 mmHg, Heart Rate 98 bpm, Respiratory Rate 18/min, SpO₂ 97% on room air.

PHYSICAL EXAMINATION: Diaphoretic. Cardiovascular exam: regular rate and rhythm, no murmurs. Lungs clear. No leg edema.

LABORATORY RESULTS: Troponin I 4.8 ng/mL (markedly elevated), CK-MB 42 U/L (elevated). BMP within normal limits.

ECG: ST elevation in leads II, III, aVF. Reciprocal changes in I, aVL.`,
  },
  {
    id: 4,
    title: "28yo Male — Chest Pain",
    meta: "Cardiac",
    text: `PATIENT: Male, 28 years old.

CHIEF COMPLAINT: Chest pain.

HISTORY OF PRESENT ILLNESS: Patient complains of chest pain since yesterday. The pain is sharp and located in the center of the chest. He is worried it might be heart-related.

PAST MEDICAL HISTORY: Nothing to report.

MEDICATIONS: None.

ALLERGIES: None known.`,
  },
];
