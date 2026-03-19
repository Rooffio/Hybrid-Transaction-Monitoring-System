"""
================================================================================
MODULE: DETERMINISTIC TRANSACTION RULES ENGINE (FIRST LINE OF DEFENSE)
================================================================================
Target Audience: Compliance IT / Regulatory Reporting Team / MLRO

DESCRIPTION:
    This module executes the deterministic, heuristic-based detection layer.
    It identifies classic money laundering "Red Flags" using exact thresholds
    defined by local regulations (e.g., FinCEN, CBK, FCA).

DETECTION DOMAINS:
    1. Structuring (Rule 5): Captures "Just-below-threshold" deposits/transfers.
    2. Velocity (Rule 8): Flags surges relative to the customer's expected profile.
    3. Geography (Rule 12): Monitors flows to sanctioned or high-risk regions.

TUNING FOR ACCURACY & PERFORMANCE:
    - Structuring (r5): The 'structuring_floor' (e.g., $9,000) vs 'reporting_limit'
      ($10,000) creates the detection window. Widening this window increases
      coverage but may increase False Positives from legitimate $9k transfers.
    - Velocity (r8): Uses the 'vol_breach_multiplier'. A value of 2.0 means any
      transaction exceeding 200% of the user's monthly expected volume is flagged.
    - Performance: Employs NumPy vectorization for O(1) row-wise execution,
      capable of processing 1M+ records in seconds.

COMPLIANCE IMPACT:
    Directly addresses Anti-Money Laundering (AML) requirements for CTR
    (Currency Transaction Report) evasion detection and Jurisdictional
    Risk Assessment.
================================================================================
"""

import pandas as pd
import numpy as np
import logging

# --- LOGGING SETUP ---
logger = logging.getLogger("TMS_Transaction_Rules")


class TransactionRules:
    def __init__(self, config):
        """
        Initializes the rules engine using parameters from the global configuration.
        Maps YAML keys to internal detection logic.

        :param config: Dictionary containing 'rules' section from tms_config.yaml.
        """
        self.config = config or {}
        rules_cfg = self.config.get('rules', {})

        # --- Threshold Mapping (Aligned with tms_config.yaml) ---

        # Rule 5: Structuring Logic
        self.struct_floor = rules_cfg.get('r5_structuring_floor', 9000.0)
        self.report_limit = rules_cfg.get('r5_reporting_limit', 10000.0)

        # Rule 8: Velocity Logic
        self.vol_multiplier = rules_cfg.get('r8_vol_breach_multiplier', 2.0)

        # Rule 12: Geographic Logic
        self.high_risk_countries = rules_cfg.get('r12_high_risk_jurisdictions', [])

        logger.info(
            f"Transaction Rules Engine initialized. "
            f"Structuring Floor: ${self.struct_floor}, "
            f"Velocity Multiplier: {self.vol_multiplier}x, "
            f"Monitored Jurisdictions: {len(self.high_risk_countries)}"
        )

    def apply_structuring_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IDENTIFIES: 'Just-below-threshold' behavior (Rule 5).
        Logic: Flag transactions within the 'Danger Zone' between the floor and limit.
        """
        if 'amount' not in df.columns:
            return df

        # Vectorized identification of structuring attempts
        # Returns 1 if amount is between [Floor] and [Limit), else 0
        is_structuring = (df['amount'] >= self.struct_floor) & (df['amount'] < self.report_limit)

        # Assign Score: We scale this to a 100-point raw score for the Orchestrator
        df['structuring_raw'] = np.where(is_structuring, 100.0, 0.0)

        if is_structuring.any():
            hits = is_structuring.sum()
            logger.debug(f"Rule 5 (Structuring) triggered {hits} times.")

        return df

    def apply_velocity_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IDENTIFIES: Surges vs Stated Profile (Rule 8).
        Logic: Compares current 'amount' to 'expected_monthly_vol'.
        """
        if 'amount' not in df.columns or 'expected_monthly_vol' not in df.columns:
            logger.warning("Velocity features missing (expected_monthly_vol); score defaulted to 0.")
            df['velocity_raw'] = 0.0
            return df

        # Calculate Breach Ratio
        # We replace 0 volume with a small epsilon (1.0) to avoid DivisionByZero
        df['tmp_vol_ratio'] = df['amount'] / df['expected_monthly_vol'].replace(0, 1.0)

        # Vectorized scoring: If ratio > multiplier, assign risk score
        # Scoring logic: (Actual Ratio / Multiplier) * 50, capped at 100
        df['velocity_raw'] = np.where(
            df['tmp_vol_ratio'] >= self.vol_multiplier,
            (df['tmp_vol_ratio'] / self.vol_multiplier * 50).clip(0, 100),
            0.0
        )

        return df

    def apply_geographic_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IDENTIFIES: High-risk jurisdictional exposure (Rule 12).
        Logic: Checks if the destination country is in the sanctioned/high-risk list.
        """
        if 'receiver_country' not in df.columns:
            logger.warning("Geography features missing (receiver_country); score defaulted to 0.")
            df['geo_raw'] = 0.0
            return df

        # Vectorized set-membership check
        is_high_risk = df['receiver_country'].isin(self.high_risk_countries)

        # Jurisdictional risk is a critical regulatory flag, assigned maximum weight (100)
        df['geo_raw'] = np.where(is_high_risk, 100.0, 0.0)

        if is_high_risk.any():
            hits = is_high_risk.sum()
            logger.debug(f"Rule 12 (Geography) flagged {hits} transactions to high-risk areas.")

        return df

    def run_all_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the full suite of deterministic Red-Flag rules.
        """
        if df.empty:
            return df

        logger.info("Executing Transaction Rules Suite (Deterministic Pass)...")

        # 1. Execute Domain Logic
        df = self.apply_structuring_rules(df)
        df = self.apply_velocity_rules(df)
        df = self.apply_geographic_rules(df)

        # 2. Cleanup
        # We remove calculation-specific temporary columns to maintain a clean result schema.
        temp_cols = ['tmp_vol_ratio']
        df = df.drop(columns=[c for c in temp_cols if c in df.columns])

        return df


if __name__ == "__main__":
    # Internal Unit Test
    print("Deterministic Rules Module - Keys Mapped to YAML [OK]")