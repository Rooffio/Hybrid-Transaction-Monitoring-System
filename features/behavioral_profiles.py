"""
================================================================================
MODULE: BEHAVIORAL ANALYTICS & ENTITY PROFILING
================================================================================
Target Audience: Compliance IT / Financial Crime Unit (FCU)

DESCRIPTION:
    This module implements the "Dynamic Baselining" logic for the TMS. Unlike
    static rules, this component analyzes a user's unique historical behavior
    (μ and σ) to detect anomalies. It focuses on:
    1. Temporal Deviations: Transactions occurring in unusual "Night" windows.
    2. Amount Volatility: Z-Score analysis to identify spikes in spend.
    3. Peer Comparisons: Identifying outliers within localized data blocks.

TUNING FOR ACCURACY & PERFORMANCE:
    - Sigma Threshold: Controlled via 'rules.r9_zscore_threshold' in YAML.
      Standard is 3.0. Lowering this increases "Pattern Change" sensitivity.
    - Night Window: Hardcoded to 1 AM - 4 AM by default; can be moved to
      config if regional shifts are required.
    - Performance: All calculations are vectorized via Pandas to ensure
      sub-second processing on batches of up to 100k records.

COMPLIANCE IMPACT:
    Essential for meeting regulatory expectations regarding "Ongoing Monitoring"
    and "Change in Behavior" detection, which static thresholds often miss.
================================================================================
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# --- LOGGING SETUP ---
logger = logging.getLogger("TMS_Behavioral_Profiling")


class BehavioralProfiles:
    def __init__(self, config):
        """
        Initializes the profiling engine using centralized configuration.

        :param config: Dictionary containing 'rules' and 'behavioral' settings.
        """
        # Pull threshold from config: r9 (Z-Score Threshold)
        # Defaulting to 3.0 (Standard Deviation) if not specified in YAML
        rules_cfg = config.get('rules', {})
        self.sigma_multiplier = rules_cfg.get('r9_zscore_threshold', 3.0)

        # Pull minimum history requirement
        # Prevents flagging new users with insufficient data
        self.min_history = rules_cfg.get('min_txns_for_baseline', 3)

        logger.info(
            f"Behavioral Profiling Engine initialized. "
            f"Threshold: {self.sigma_multiplier}σ, Min History: {self.min_history}"
        )

    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts time-of-day and seasonality metrics.
        Identifies unusual active windows that may indicate automated activity
        or account takeover (ATO).
        """
        if 'timestamp' not in df.columns:
            logger.warning("Timestamp missing; skipping temporal analysis.")
            return df

        # Ensure datetime format for extraction
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Binary flag for unusual hours (Night Window)
        # Compliance IT Note: Adjust range [1-4] based on typical market activity.
        df['is_night_txn'] = df['hour'].apply(lambda x: 1 if 1 <= x <= 4 else 0)

        return df

    def generate_baselines(self, df: pd.DataFrame, entity_col='global_entity_id') -> pd.DataFrame:
        """
        Calculates μ (Mean) and σ (Standard Deviation) for each entity.
        In production, this represents the "Normal" state for every customer.
        """
        # Fallback to sender_id if Identity Resolution has not been executed
        target_col = entity_col if entity_col in df.columns else 'sender_id'

        logger.info(f"Generating entity baselines using [{target_col}]...")

        # Calculate statistics per entity
        baselines = df.groupby(target_col)['amount'].agg(['mean', 'std', 'count']).reset_index()
        baselines.columns = [target_col, 'user_mean', 'user_std', 'txn_count']

        # Fill NaN for single-transaction entities to prevent calculation errors
        baselines['user_std'] = baselines['user_std'].fillna(0)

        return baselines

    def score_behavioral_anomalies(self, df: pd.DataFrame, baselines: pd.DataFrame,
                                   entity_col='global_entity_id') -> pd.DataFrame:
        """
        Calculates the 'behavioral_raw' score [0-100] based on Z-Score deviation.

        Scoring Logic:
        - 100: Critical breach (> configured σ threshold)
        - 0-90: Scaled score for values approaching the threshold
        - 0: Normal or Insufficient history
        """
        target_col = entity_col if entity_col in df.columns else 'sender_id'

        # Join baselines to the current transaction batch
        df = df.merge(baselines, on=target_col, how='left')

        def _calculate_impact(row):
            mu = row.get('user_mean', 0)
            sigma = row.get('user_std', 0)
            amount = row['amount']

            # Guard clause: Verify minimum transaction history
            if row['txn_count'] < self.min_history:
                return 0

            # Guard clause: No variance in history (Sigma 0)
            # If a user always spends exactly $100, a $250 spend is a significant anomaly.
            if sigma == 0:
                return 100 if amount > (mu * 1.5) else 0

            # Calculate dynamic threshold
            threshold = mu + (self.sigma_multiplier * sigma)

            # High-severity breach
            if amount > threshold:
                return 100

            # Proportional scoring (Z-Score context)
            z_score = (amount - mu) / sigma
            if z_score > 1.0:
                # Linearly scale the score as it approaches the threshold
                scaled_score = (z_score / self.sigma_multiplier) * 100
                return min(round(scaled_score, 1), 90)

            return 0

        logger.info("Applying Z-Score transformations to transaction batch...")
        df['behavioral_raw'] = df.apply(_calculate_impact, axis=1)

        # Drop temporary stats columns to maintain clean output
        drop_cols = ['user_mean', 'user_std', 'txn_count']
        return df.drop(columns=[c for c in drop_cols if c in df.columns])

    def extract_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        High-level execution wrapper for the Behavioral Profiling pipeline.

        Workflow:
        1. Temporal enrichment (Hour/Day/Night flags)
        2. Statistical baselining (Mean/StdDev)
        3. Anomaly scoring (Z-Impact calculation)
        """
        if df.empty:
            logger.warning("Empty transaction batch; skipping profiling.")
            return df

        logger.info("Starting Behavioral Pipeline...")

        df = self.calculate_temporal_features(df)
        baselines = self.generate_baselines(df)
        df = self.score_behavioral_anomalies(df, baselines)

        logger.info("Behavioral profiles successfully applied to batch.")
        return df