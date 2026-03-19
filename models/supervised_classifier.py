"""
================================================================================
MODULE: SUPERVISED ML CLASSIFIER (XGBOOST SAR PREDICTION)
================================================================================
Target Audience: Compliance IT / Data Science Engineering

DESCRIPTION:
    This module implements "Supervised Learning" to identify patterns seen in
    historical Suspicious Activity Reports (SARs). While rules look for specific
    indicators, this model looks for the *combination* of features that
    historically led to a SAR filing.

    It utilizes XGBoost (Extreme Gradient Boosting), which is the industry
    standard for tabular financial data due to its handling of non-linear
    relationships and missing values.

PERFORMANCE & TUNING:
    - Model Hyperparameters: Controlled via 'ml_models' in tms_config.yaml.
      Tuning 'max_depth' and 'learning_rate' allows for a trade-off between
      model complexity and generalization.
    - Probability Scaling: The raw [0, 1] probability is scaled to [0, 100]
      to maintain consistency with the Risk Orchestrator's scoring logic.
    - Feature Consistency: The module ensures that the training and inference
      schemas remain identical via the '_prepare_features' wrapper.

COMPLIANCE IMPACT:
    Automates the "Historical Comparison" requirement of AML regulations.
    By training on past confirmed cases of money laundering, the system
    effectively "learns" from the institution's own historical expertise.
================================================================================
"""

import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import os

# --- LOGGING SETUP ---
logger = logging.getLogger("TMS_Supervised_Classifier")


class SupervisedClassifier:
    def __init__(self, config=None):
        """
        Initializes the classifier using parameters from the central config.

        :param config: Dictionary containing 'ml_models' settings from YAML.
        """
        self.config = config or {}
        ml_cfg = self.config.get('ml_models', {})

        # Path management
        self.model_path = ml_cfg.get('model_artifact_path', 'data/cache/xgboost_sar_model.joblib')
        self.model = None

        # Training Hyperparameters
        self.max_depth = ml_cfg.get('max_depth', 6)
        self.learning_rate = ml_cfg.get('learning_rate', 0.1)
        self.n_estimators = ml_cfg.get('n_estimators', 100)

        logger.info(f"Supervised Engine initialized. Artifact Target: {self.model_path}")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters and aligns features. This is the "Contract" between the ETL
        and the ML model. Any feature used in training MUST be present here.
        """
        feature_cols = [
            'amount',
            'account_age_days',
            'user_risk_score',
            'historical_avg_txn',
            'historical_std_dev',
            'expected_monthly_vol',
            'known_counterparties'
        ]

        # Ensure schema stability
        X = df.copy()
        for col in feature_cols:
            if col not in X.columns:
                # If a feature is missing from the batch, we log a warning and
                # default to 0 to prevent a model crash.
                logger.warning(f"Feature '{col}' missing from batch. Imputing 0.")
                X[col] = 0

        # Return aligned feature set with NaN handling
        return X[feature_cols].fillna(0)

    def train_model(self, df: pd.DataFrame, label_col: str = 'is_sar_filed'):
        """
        Performs Supervised Training using historical labeled data.
        Saves a binary artifact (joblib) for use in the production pipeline.
        """
        if label_col not in df.columns:
            logger.error(f"Training failed: Label column '{label_col}' not found.")
            return

        logger.info(f"Training on {len(df)} historical SAR/Non-SAR samples...")

        X = self._prepare_features(df)
        y = df[label_col].astype(int)

        # Standard 80/20 split for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize XGBoost with specific AML configurations
        # Logic: binary:logistic produces a probability, which we need for scoring.
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

        logger.info("Fitting XGBoost model...")
        self.model.fit(X_train, y_train)

        # Audit check: Log validation performance
        accuracy = self.model.score(X_val, y_val)
        logger.info(f"Training complete. Validation Accuracy: {accuracy:.4f}")

        # Persistence: Save the binary artifact
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model artifact saved to: {self.model_path}")

    def predict_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes inference on a live batch of transactions.
        If no model exists, it safely returns 0 scores to allow the pipeline to continue.
        """
        # Lazy Loading: Load model into memory only when needed
        if self.model is None:
            if os.path.exists(self.model_path):
                try:
                    self.model = joblib.load(self.model_path)
                    logger.info("Successfully loaded SAR prediction model from disk.")
                except Exception as e:
                    logger.error(f"Critical error loading ML artifact: {e}")
            else:
                logger.warning("No SAR prediction model found. ML-Supervised scores will be 0.")
                df['ml_supervised_score'] = 0.0
                return df

        # Align features
        X = self._prepare_features(df)

        # Predict probabilities [Non-SAR, SAR]
        # We extract the second column (SAR probability)
        probabilities = self.model.predict_proba(X)[:, 1]

        # Scale to [0-100] for Risk Orchestrator compatibility
        df['ml_supervised_score'] = (probabilities * 100).round(2)

        logger.info(f"Supervised scoring complete. Max Prob: {df['ml_supervised_score'].max():.1f}")
        return df


if __name__ == "__main__":
    # Internal component state check
    print("XGBoost Supervised Classification Module: READY")