"""
================================================================================
MODULE: STATE MANAGER & PERSISTENCE ORCHESTRATOR
================================================================================
Target Audience: Compliance IT / Financial Crime Unit (FCU)

DESCRIPTION:
    This module manages the "Memory" of the Transaction Monitoring System (TMS).
    It handles two critical types of state:
    1. Delta Checkpoints: Tracks the last processed timestamp (Watermark) to
       ensure no transaction is ever processed twice.
    2. Behavioral Baselines: Persists historical statistics (Mean/StdDev) for
       entities, allowing the system to detect "Change in Pattern" without
       re-reading the entire historical database for every run.

KEY FEATURES:
    - Atomic State Updates: Metadata is saved in JSON format for auditability.
    - Configuration-Driven: Paths for cache and metadata are pulled from
      tms_config.yaml to ensure environmental consistency.
    - Delta Filtering: High-performance Pandas-based filtering of new records.

PERFORMANCE & TUNING:
    - Cache Location: Controlled via 'storage.cache_dir'. For high-availability
      production, this can be pointed to a shared network drive or an S3 mount.
    - Versioning: 'version' in metadata allows for schema evolution if the
      risk scoring logic changes significantly.

COMPLIANCE IMPACT:
    Ensures data integrity and prevents "Alert Fatigue" caused by re-processing
    stale data. Provides a clear audit trail of when the system last ran and
    how many records were ingested.
================================================================================
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime

# --- LOGGING SETUP ---
logger = logging.getLogger("TMS_StateManager")


class StateManager:
    def __init__(self, config=None):
        """
        Initializes the state management engine.

        :param config: Dictionary containing 'storage' settings from tms_config.yaml.
                       Defaults to standard paths if config is not provided.
        """
        # Extract storage parameters from central config
        storage_cfg = config.get('storage', {}) if config else {}

        self.cache_dir = storage_cfg.get('cache_dir', "data/cache")
        self.metadata_file = storage_cfg.get('metadata_file', "last_run_metadata.json")
        self.metadata_path = os.path.join(self.cache_dir, self.metadata_file)

        # Ensure directory structure exists for persistence
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load existing state into memory
        self.state = self._load_state()

        logger.info(f"StateManager initialized. Monitoring watermark at: {self.metadata_path}")

    def _load_state(self):
        """
        Loads the system's "Last Run" context from disk.

        Compliance Note: If the file is missing, the system defaults to 1970
        (Epoch), essentially triggering a full historical re-scan.
        """
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Integrity Error: Failed to parse state metadata: {e}")

        # Default state: Initialized for a "First Run" scenario
        return {
            "last_run_timestamp": "1970-01-01T00:00:00",
            "total_processed_to_date": 0,
            "version": "2.5.0",  # Aligned with Hybrid Engine Version
            "last_updated": None
        }

    def get_delta(self, df: pd.DataFrame, timestamp_col='timestamp') -> pd.DataFrame:
        """
        Filters out already-processed transactions based on the saved watermark.

        Logic:
            Only rows where timestamp > last_run_timestamp are returned.

        Performance:
            Vectorized comparison in Pandas for 1M+ rows.
        """
        if df.empty:
            return df

        # Ensure timestamp column is strictly typed for comparison
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        last_run = pd.to_datetime(self.state["last_run_timestamp"])

        new_records = df[df[timestamp_col] > last_run].copy()

        if not new_records.empty:
            logger.info(
                f"Delta identified: {len(new_records)} new records found. "
                f"Previous watermark: {last_run.isoformat()}."
            )
        else:
            logger.warning("No new transactions found since the last successful execution.")

        return new_records

    def save_state(self, df: pd.DataFrame, timestamp_col='timestamp'):
        """
        Advances the watermark and persists the updated context.

        Important: This must only be called at the end of the pipeline
        to ensure atomicity (i.e., we only advance the watermark if
        processing was successful).
        """
        if df.empty:
            logger.info("Empty batch processed; watermark remains unchanged.")
            return

        # Advance the watermark to the most recent transaction in this batch
        latest_timestamp = df[timestamp_col].max()

        # Format for JSON serialization
        if isinstance(latest_timestamp, pd.Timestamp):
            latest_timestamp_str = latest_timestamp.isoformat()
        else:
            latest_timestamp_str = str(latest_timestamp)

        # Update in-memory state
        self.state["last_run_timestamp"] = latest_timestamp_str
        self.state["total_processed_to_date"] += len(df)
        self.state["last_updated"] = datetime.now().isoformat()

        # Persist to disk
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.state, f, indent=4)
            logger.info(f"System state persisted. New Watermark: {latest_timestamp_str}")
        except Exception as e:
            logger.critical(f"Storage Failure: Could not update watermark: {e}")

    def save_rolling_stats(self, feature_name: str, stats_dict: dict):
        """
        Persists derived behavioral features (e.g., User Averages).
        Used by the BehavioralProfiles module to avoid recalculating
        normals from scratch every time.
        """
        stats_path = os.path.join(self.cache_dir, f"stats_{feature_name}.json")
        try:
            with open(stats_path, 'w') as f:
                json.dump(stats_dict, f, indent=4)
            logger.debug(f"Rolling stats for [{feature_name}] updated successfully.")
        except Exception as e:
            logger.error(f"Feature Cache Error: Failed to save stats for {feature_name}: {e}")

    def load_rolling_stats(self, feature_name: str) -> dict:
        """
        Retrieves cached behavioral statistics. Returns None if no
        baseline exists for this feature yet.
        """
        stats_path = os.path.join(self.cache_dir, f"stats_{feature_name}.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Read Error: Could not load stats for {feature_name}: {e}")
        return None

    def reset_state(self):
        """
        Administrative Utility: Deletes current state to trigger
        a full historical re-processing of the input data.
        """
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
            logger.warning("System state has been MANUALLY RESET. Full re-scan will occur on next run.")
        self.state = self._load_state()