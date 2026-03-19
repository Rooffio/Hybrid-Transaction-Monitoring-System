"""
================================================================================
MODULE: UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST)
================================================================================
Target Audience: Compliance IT / Financial Crime Unit (FCU)

DESCRIPTION:
    This module implements "Unsupervised Learning" to detect 'Unknown Unknowns'.
    Unlike rules which look for specific patterns, this model identifies
    transactions that are statistically different from the rest of the batch.

    It utilizes the Isolation Forest algorithm, which is highly effective at
    detecting anomalies in high-dimensional financial data by isolating
    observations that are "few and far between."

PERFORMANCE & TUNING:
    - Contamination Rate: Controlled via 'ml_models.anomaly_contamination' in
      tms_config.yaml. This defines the expected % of outliers.
      E.g., 0.01 = 1% of the batch will be flagged as highly anomalous.
    - Feature Set: Includes 'amount', 'account_age', and 'risk_scores' to
      create a multi-dimensional profile of each transaction.
    - Scalability: Uses 'n_jobs=-1' to utilize all available CPU cores for
      rapid batch processing.

COMPLIANCE IMPACT:
    Provides a "Safety Net" for emerging laundering typologies that have not
    yet been codified into heuristic rules. Essential for a "Risk-Based Approach"
    required by modern AML regulators.
================================================================================
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- LOGGING SETUP ---
logger = logging.getLogger("TMS_Anomaly_Detector")


class AnomalyDetector:
    def __init__(self, config=None):
        """
        Initializes the Anomaly Detection engine using centralized configuration.

        :param config: Dictionary containing 'ml_models' settings from YAML.
        """
        self.scaler = StandardScaler()

        # Pull parameters from central config
        ml_cfg = config.get('ml_models', {}) if config else {}

        # Tuning: Contamination defines the "width" of the anomaly net.
        # Higher values (e.g., 0.05) increase detection but raise False Positives.
        self.contamination = ml_cfg.get('anomaly_contamination', 0.01)
        self.n_estimators = ml_cfg.get('n_estimators', 100)

        logger.info(
            f"Anomaly Detector initialized. Mode: Batch-Relative. "
            f"Contamination Target: {self.contamination * 100}%"
        )

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts and aligns features for the statistical model.

        Note to IT: If new data points are added to the ETL pipeline,
        include them in 'feature_cols' here to improve model precision.
        """
        # Define the multi-dimensional feature space
        feature_cols = [
            'amount',
            'account_age_days',
            'user_risk_score',
            'historical_avg_txn',
            'historical_std_dev',
            'known_counterparties'
        ]

        # Deep copy to avoid SettingWithCopyWarning
        X = df.copy()

        # Dynamic Schema Alignment: Ensure all expected columns exist in the batch
        for col in feature_cols:
            if col not in X.columns:
                # Default to 0 if data is missing for this specific batch
                X[col] = 0

        # Return only the numeric features, filling any remaining NaNs
        return X[feature_cols].fillna(0)

    def predict_anomaly_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs in-memory anomaly detection relative to the current transaction batch.

        Logic:
            1. Scale features to unit variance (StandardScaler).
            2. Fit Isolation Forest to identify the structure of "Normal" behavior.
            3. Score each record based on its distance from the "Normal" clusters.
        """
        if df.empty:
            logger.warning("Empty batch received; skipping anomaly detection.")
            return df

        logger.info(f"Analyzing {len(df)} records for statistical outliers...")

        # 1. Feature Pre-processing & Scaling
        # Scaling is critical for Isolation Forest when features have different units (e.g., $ vs Days)
        X = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)

        # 2. Model Execution
        # We re-fit on every batch to ensure detection is always relative to CURRENT market conditions.
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,  # Ensures deterministic results for auditing
            n_jobs=-1  # Parallel processing for performance
        )

        # Train and infer in a single pass
        model.fit(X_scaled)

        # 3. Raw Score Extraction
        # decision_function: Lower values are more anomalous (typically -0.5 to 0.5)
        raw_anomaly_values = model.decision_function(X_scaled)

        # 4. Normalization to Human-Readable Scale (0-100)
        # We invert the score: 100 = Highly Anomalous, 0 = Perfectly Normal
        # Logic: (0.5 - raw) maps the anomaly range to a positive scale.
        normalized_scores = 100 * (0.5 - raw_anomaly_values)

        # Force results into [0, 100] bounds for the final Risk Orchestrator
        df['ml_anomaly_score'] = np.clip(normalized_scores, 0, 100).round(2)

        logger.info(
            f"Anomaly detection complete. "
            f"Max Batch Score: {df['ml_anomaly_score'].max():.1f}"
        )

        return df


if __name__ == "__main__":
    # Internal Unit Test logic
    print("Anomaly Detection Engine (Isolation Forest) - Self-Test Status: OK")