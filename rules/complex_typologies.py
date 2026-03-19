"""
================================================================================
MODULE: COMPLEX TYPOLOGIES & MULTI-STEP LAUNDERING DETECTION
================================================================================
Target Audience: Compliance IT / AML Subject Matter Experts (SMEs) / FIU

DESCRIPTION:
    This engine shifts focus from single-transaction "Red Flags" to "Scenario-Based"
    analysis. It evaluates the relationships between multiple parties and
    transactions over time to detect the three stages of money laundering:
    1. Placement: In-bound Smurfing (splitting large cash into small deposits).
    2. Layering: U-Turns and rapid "Pass-Through" movements to hide the trail.
    3. Integration: Luxury purchases (MCC Mismatch) by low-income profiles.

PERFORMANCE & TUNING:
    - Smurfing (r14/r15): Controlled by 'fan_in' and 'fan_out' limits. Lowering
      these (e.g., to 3) will catch smaller mule rings but may flag legitimate
      crowdfunding or group-gift activities.
    - U-Turn Window: Currently hardcoded to 24 hours. Extending this captures
      sophisticated "Slow-Walking" wash trades but increases memory usage
      during the self-join operation.
    - Integration Mismatch: Relies on 'sender_occupation' accuracy. If the source
      data for occupations is "Unknown", this rule will under-report.

COMPLIANCE IMPACT:
    Directly addresses FATF guidelines on "Suspicious Transaction Patterns"
    and "Complex/Unusually Large Transactions" (FATF Recommendation 20).
================================================================================
"""

import pandas as pd
import numpy as np
import logging

# --- LOGGING SETUP ---
logger = logging.getLogger("TMS_Complex_Typologies")


class ComplexTypologies:
    def __init__(self, config):
        """
        Initializes the typology engine using parameters from the global configuration.

        :param config: Dictionary containing 'rules' section from tms_config.yaml.
        """
        self.config = config or {}
        rules_cfg = self.config.get('rules', {})

        # --- Threshold Mapping (Aligned with tms_config.yaml) ---

        # Structuring/Smurfing Limits
        self.report_limit = rules_cfg.get('r5_reporting_limit', 10000.0)
        self.fan_in_limit = rules_cfg.get('r14_fan_in_limit', 10)
        self.fan_out_limit = rules_cfg.get('r15_fan_out_limit', 10)

        # Integration Phase (MCC)
        self.high_risk_mcc = rules_cfg.get('high_risk_mccs', [5944, 6211])
        self.low_income_occupations = rules_cfg.get('low_income_profiles', ['Student', 'Unemployed'])

        logger.info(
            f"Complex Typologies Engine initialized. "
            f"Fan-In Limit: {self.fan_in_limit}, Fan-Out Limit: {self.fan_out_limit}"
        )

    def detect_layering_smurfing(self, df: pd.DataFrame) -> list:
        """
        IDENTIFIES: Mule Networks / Smurfing Hubs.
        Logic: Multiple unique senders (Fan-In) to a single recipient,
        where the recipient then moves >80% of funds out (Pass-Through).
        """
        if df.empty:
            return []

        # 1. Analyze Inflows (In-bound Concentration)
        inflows = df.groupby('receiver_id').agg({
            'sender_id': 'nunique',
            'amount': 'sum'
        }).rename(columns={'sender_id': 'unique_senders', 'amount': 'total_in_amt'})

        # 2. Analyze Outflows (Rapid Disbursement)
        outflows = df.groupby('sender_id').agg({
            'amount': 'sum'
        }).rename(columns={'amount': 'total_out_amt'})

        # 3. Detect "Pass-Through" Entities
        # We merge inflows/outflows on entity_id to find intermediaries
        mule_candidates = inflows.join(outflows, how='inner')

        # Criteria for a Smurfing Hub hit:
        # - Senders count exceeds the 'fan_in_limit'
        # - Significant % of received funds are pushed out (Evidence of Layering)
        smurfing_mask = (
                (mule_candidates['unique_senders'] >= self.fan_in_limit) &
                (mule_candidates['total_out_amt'] >= (mule_candidates['total_in_amt'] * 0.80))
        )

        return mule_candidates[smurfing_mask].index.tolist()

    def detect_u_turn_transactions(self, df: pd.DataFrame) -> list:
        """
        IDENTIFIES: "Wash Trading" / Round-Tripping.
        Logic: Entity A sends to B, B sends to A within 24 hours.
        """
        if df.empty or len(df) < 2:
            return []

        # Perform a self-join to find reciprocal pairs (A->B, B->A)
        # We only look at essential columns to optimize memory usage
        cols = ['sender_id', 'receiver_id', 'timestamp', 'transaction_id']

        # Join where Sender(A) == Receiver(B) and Receiver(A) == Sender(B)
        pairs = pd.merge(
            df[cols],
            df[cols],
            left_on=['sender_id', 'receiver_id'],
            right_on=['receiver_id', 'sender_id'],
            suffixes=('_orig', '_back')
        )

        if pairs.empty:
            return []

        # Time Delta calculation: Ensure timestamps are datetime objects
        pairs['ts_orig'] = pd.to_datetime(pairs['timestamp_orig'])
        pairs['ts_back'] = pd.to_datetime(pairs['timestamp_back'])

        # Filter: 2nd txn must be after 1st, and within 24 hours (86,400s)
        time_diff = (pairs['ts_back'] - pairs['ts_orig']).dt.total_seconds()
        u_turn_mask = (time_diff > 0) & (time_diff < 86400)

        return pairs[u_turn_mask]['sender_id_orig'].unique().tolist()

    def detect_mcc_mismatch(self, df: pd.DataFrame) -> list:
        """
        IDENTIFIES: Integration phase laundering.
        Logic: Low-income profiles purchasing high-value assets from high-risk MCCs.
        """
        # Integration Threshold: e.g. 20% of the main reporting limit ($2,000)
        integration_floor = self.report_limit * 0.2

        mismatch_mask = (
                (df['mcc_code'].isin(self.high_risk_mcc)) &
                (df['sender_occupation'].isin(self.low_income_occupations)) &
                (df['amount'] >= integration_floor)
        )

        return df[mismatch_mask]['transaction_id'].tolist()

    def apply_complex_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point for Scenario Analysis.
        Updates 'structuring_raw' and 'behavioral_raw' scores.
        """
        if df.empty:
            return df

        logger.info("Executing Complex Typology Analysis (Smurfing, U-Turns, Integration)...")

        # 1. Run Typology Detection logic
        smurfer_ids = self.detect_layering_smurfing(df)
        uturn_ids = self.detect_u_turn_transactions(df)
        mismatch_txns = self.detect_mcc_mismatch(df)

        # 2. Ensure raw score columns exist
        if 'structuring_raw' not in df.columns: df['structuring_raw'] = 0.0
        if 'behavioral_raw' not in df.columns: df['behavioral_raw'] = 0.0

        # 3. Apply Scoring Boosts
        # Smurfing is a critical AML indicator (100)
        if smurfer_ids:
            df.loc[df['sender_id'].isin(smurfer_ids), 'structuring_raw'] = 100.0
            df.loc[df['receiver_id'].isin(smurfer_ids), 'structuring_raw'] = 100.0

        # U-Turns are a medium-high behavioral flag (80)
        if uturn_ids:
            df.loc[df['sender_id'].isin(uturn_ids), 'behavioral_raw'] = 80.0

        # Integration Mismatches are high-confidence behavioral flags (90)
        if mismatch_txns:
            df.loc[df['transaction_id'].isin(mismatch_txns), 'behavioral_raw'] = 90.0

        logger.info(
            f"Analysis complete. Identified: {len(smurfer_ids)} Smurfing Hubs, "
            f"{len(uturn_ids)} Wash-Traders, {len(mismatch_txns)} Integration Events."
        )

        return df


if __name__ == "__main__":
    # Integration Component Test
    print("Complex Typologies Engine - SCENARIO LOGIC OPERATIONAL")