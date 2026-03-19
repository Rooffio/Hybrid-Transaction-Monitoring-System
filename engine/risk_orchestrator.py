"""
================================================================================
MODULE: RISK ORCHESTRATOR & BEHAVIORAL ANALYTICS
================================================================================
Target Audience: Compliance IT / Financial Crime Unit (FCU) Engineering

DESCRIPTION:
    This module serves as the central "Processing Brain" of the TMS. It performs
    two critical functions:
    1. Behavioral Profiling: Calculates Z-Scores and temporal anomalies (Night Txns)
       based on the user's historical or batch-relative mean and standard deviation.
    2. Hybrid Orchestration: Aggregates signals from Deterministic Rules,
       Unsupervised Anomalies, and Supervised ML into a single 0-100 Risk Score.

SCORING LOGIC & TUNING:
    - Z-Score Threshold (r9): Controls sensitivity to volume spikes. A 3.0σ
      threshold (default) follows the Three-Sigma Rule, flagging only the top 0.3%
      of deviations. Lowering this increases the "Behavioral" hit rate.
    - Weighted Aggregation: Uses the 'weights' block in tms_config.yaml.
      Total influence = (Rules * W1) + (Anomaly * W2) + (Supervised * W3).
    - Accuracy Fix: The orchestrator now dynamically finds all columns ending
      in '_raw' to ensure no rule-based signal is lost during aggregation.

COMPLIANCE IMPACT:
    Supports the "Risk-Based Approach" (RBA) required by FATF and FinCEN.
    By consolidating multiple risk vectors, it identifies high-risk clusters
    that single-rule systems would miss, significantly reducing False Positives.
================================================================================
"""

import pandas as pd
import numpy as np
import logging

# --- LOGGING SETUP ---
logger = logging.getLogger("TMS_Risk_Orchestrator")

class BehavioralProfiles:
    def __init__(self, config):
        """
        Initializes the behavioral engine using parameters from the global config.
        """
        self.config = config or {}
        rules_cfg = self.config.get('rules', {})

        # Rule 9: Z-Score Threshold (Standard Deviations)
        # Tuning: Lowering this (e.g., to 2.0) makes the system much more sensitive
        # to minor spending spikes compared to a user's average.
        self.sigma_multiplier = rules_cfg.get('r9_zscore_threshold', 3.0)

        logger.info(f"Behavioral Engine initialized with {self.sigma_multiplier}σ threshold.")

    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies "Unusual Active Windows" (Rule 11/12 Context).
        Flags transactions occurring in the 1AM-4AM window, which is often
        correlated with automated script activity or international mule timing.
        """
        if 'timestamp' not in df.columns:
            return df

        # Ensure timestamp is datetime for extraction
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['tmp_hour'] = df['timestamp'].dt.hour

        # Vectorized flag: 1 if within midnight-4AM window
        df['is_night_txn'] = np.where((df['tmp_hour'] >= 1) & (df['tmp_hour'] <= 4), 1, 0)

        return df.drop(columns=['tmp_hour'])

    def generate_baselines(self, df: pd.DataFrame, entity_col='global_entity_id') -> pd.DataFrame:
        """
        Calculates μ (mean) and σ (std dev) for the current batch.
        In production, this would typically interface with a State Manager to
        incorporate 90 days of historical data.
        """
        target_col = entity_col if entity_col in df.columns else 'sender_id'

        logger.info(f"Generating entity-level baselines on: {target_col}")

        # Group by entity to establish "Normal" behavior for this period
        baselines = df.groupby(target_col)['amount'].agg(['mean', 'std', 'count']).reset_index()
        baselines.columns = [target_col, 'user_mean', 'user_std', 'txn_count']

        # Standard deviation for single-txn entities will be NaN; fill with 0
        baselines['user_std'] = baselines['user_std'].fillna(0)

        return baselines

    def score_behavioral_anomalies(self, df: pd.DataFrame, baselines: pd.DataFrame, entity_col='global_entity_id') -> pd.DataFrame:
        """
        Calculates the deviation from baseline and returns 'behavioral_raw' [0-100].
        Formula: Score = 100 if Z-Score >= Configured Threshold.
        """
        target_col = entity_col if entity_col in df.columns else 'sender_id'
        df = df.merge(baselines, on=target_col, how='left')

        # Vectorized Z-Score calculation with 0-division safety
        df['z_score'] = (df['amount'] - df['user_mean']) / df['user_std'].replace(0, 1.0)

        # Scoring Logic:
        # 1. Insufficient History (< 3 txns) -> 0 (Avoids cold-start noise)
        # 2. Hard Breach (>= Sigma Threshold) -> 100
        # 3. Near Breach (>= 50% of Threshold) -> Linear scaling to 100
        conditions = [
            (df['txn_count'] < 3),
            (df['z_score'] >= self.sigma_multiplier),
            (df['z_score'] >= self.sigma_multiplier * 0.5)
        ]

        choices = [
            0,
            100,
            (df['z_score'] / self.sigma_multiplier) * 100
        ]

        df['behavioral_raw'] = np.select(conditions, choices, default=0).clip(0, 100)

        return df.drop(columns=['user_mean', 'user_std', 'txn_count', 'z_score'])

class RiskOrchestrator:
    def __init__(self, config):
        """
        Initializes the Orchestrator with weighting logic from tms_config.yaml.
        Impact: Adjusting weights shifts the focus between Rules and ML models.
        """
        self.config = config or {}

        # --- Weight Mapping (Defaulting to 40/35/25 split) ---
        w_cfg = self.config.get('weights', {})
        self.rule_w = w_cfg.get('rule_weight', 0.40)
        self.anomaly_w = w_cfg.get('anomaly_weight', 0.35)
        self.super_w = w_cfg.get('supervised_weight', 0.25)

        # --- Risk Banding Thresholds ---
        b_cfg = self.config.get('risk_bands', {})
        self.crit_threshold = b_cfg.get('critical', 80)
        self.high_threshold = b_cfg.get('high', 60)
        self.med_threshold = b_cfg.get('medium', 30)

        # Initialize the Behavioral Sub-Engine
        self.behavioral_engine = BehavioralProfiles(self.config)

        logger.info(f"Orchestrator Weights: Rules({self.rule_w}), Anomaly({self.anomaly_w}), Supervised({self.super_w})")

    def calculate_final_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core logic for hybrid score aggregation.
        Ensures that multiple high-risk signals compound into a single score.
        """
        if df.empty:
            return df

        # --- 1. Behavioral Enrichment ---
        df = self.behavioral_engine.calculate_temporal_features(df)
        baselines = self.behavioral_engine.generate_baselines(df)
        df = self.behavioral_engine.score_behavioral_anomalies(df, baselines)

        # --- 2. Dynamic Rule Aggregation (Accuracy Fix) ---
        # Instead of hardcoding rule names, we catch every column ending in '_raw'.
        # This captures Structuring, Velocity, Geo-Risk, and Behavioral.
        rule_cols = [c for c in df.columns if c.endswith('_raw')]

        if rule_cols:
            # We take the maximum of all individual rule signals.
            # This follows a "Conservative Hit" approach: any 100% hit should count as a 100% rule risk.
            df['norm_rule_score'] = df[rule_cols].max(axis=1)
            logger.info(f"Aggregating {len(rule_cols)} signals: {rule_cols}")
        else:
            logger.warning("No rule columns (_raw) detected. Risk may be understated.")
            df['norm_rule_score'] = 0.0

        # --- 3. ML Component Normalization ---
        # Scales 0.0-1.0 probabilities to a 0-100 integer scale if necessary.
        for ml_col in ['ml_anomaly_score', 'ml_supervised_score']:
            if ml_col in df.columns:
                if df[ml_col].max() <= 1.0:
                    df[ml_col] = df[ml_col] * 100
            else:
                df[ml_col] = 0.0

        # --- 4. Weighted Aggregation Formula ---
        # final_score = (RuleMax * W1) + (Unsupervised * W2) + (Supervised * W3)
        df['final_risk_score'] = (
            (df['norm_rule_score'] * self.rule_w) +
            (df['ml_anomaly_score'] * self.anomaly_w) +
            (df['ml_supervised_score'] * self.super_w)
        ).round(2).clip(0, 100)

        # --- 5. Risk Prioritization (Banding) ---
        conditions = [
            (df['final_risk_score'] >= self.crit_threshold),
            (df['final_risk_score'] >= self.high_threshold),
            (df['final_risk_score'] >= self.med_threshold)
        ]
        choices = ["CRITICAL", "HIGH", "MEDIUM"]
        df['risk_band'] = np.select(conditions, choices, default="LOW")

        # Cleanup internal scoring helper columns
        drop_cols = ['norm_rule_score']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        logger.info(f"Risk aggregation complete. Max Risk Band: {df['risk_band'].value_counts().to_dict()}")
        return df

if __name__ == "__main__":
    print("Risk Orchestrator Module - HYBRID LOGIC OPERATIONAL")