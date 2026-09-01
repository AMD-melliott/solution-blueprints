// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

export type Decision = "NOT_FRAUD" | "REVIEW" | "FRAUD";
export type StreamState = "idle" | "running" | "paused" | "stopped";
export type TabId = "warmup" | "live";
export type RiskLevel = "HIGH" | "MEDIUM" | "LOW";
export type RuleAction = "BLOCK" | "REVIEW" | "MONITOR";

export interface TriggeredRule {
  rule_id: string;
  description: string;
  risk_level: RiskLevel;
  action: RuleAction;
}

export interface ScoredTransaction {
  transaction_id: number;
  transaction_dt: number | null;
  transaction_amt: number;
  card_label: string;
  score: number;
  decision: Decision;
  latency_ms: number;
  is_fraud_gt: number | null;
  ground_truth_match: boolean | null;
  product?: string | null;
  card_type?: string | null;
  billing_region_code?: number | null;
  billing_country_code?: number | null;
  email_domain?: string | null;
  device_type?: string | null;
  device_info?: string | null;
  explainability?: {
    triggered_rules: TriggeredRule[];
  } | null;
}

export interface Metrics {
  total: number;
  clean: number;
  review: number;
  fraud: number;
  latency_min_ms: number | null;
  latency_avg_ms: number | null;
  latency_max_ms: number | null;
  accuracy_pct: number | null;
  missed_fraud_pct: number | null;
  false_positive_pct: number | null;
  missed_fraud_usd: number;
  false_positive_usd: number;
}

export interface Thresholds {
  review: number;
  fraud: number;
}

export interface StreamStatus {
  state: StreamState;
  position: number;
  thresholds: Thresholds;
  metrics: Metrics;
}

export interface WarmupChecks {
  dataset: boolean;
  // Absent in fake scoring mode (no backend to probe)
  backend?: boolean;
  gnn?: boolean;
  xgb?: boolean;
}

export interface WarmupStatus {
  ready: boolean;
  scorer_mode: "real" | "fake" | null;
  dataset_rows: number;
  checks: WarmupChecks;
}

export interface FraudDetail {
  transaction_id: number;
  timestamp: string;
  amount: number;
  card_label: string;
  latency_ms: number;
  score: number;
  decision: Decision;
  ground_truth: string;
  fraud_threshold: number;
  product: string;
  card_type: string;
  billing_region_code: string;
  billing_country_code: string;
  email_domain: string;
  device_type: string;
  device_info: string;
  triggered_rules: TriggeredRule[];
}
