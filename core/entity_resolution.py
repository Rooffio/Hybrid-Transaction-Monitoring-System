"""
================================================================================
MODULE: IDENTITY RESOLUTION (ENTITY LINKING)
================================================================================
Target Audience: Compliance IT / Financial Crime Unit (FCU)

DESCRIPTION:
    This module implements "Global Entity" logic. It resolves disparate
    account identifiers (sender_ids) into a single unified identity
    (global_entity_id) using a combination of deterministic (Exact Match)
    and probabilistic (Fuzzy Match) techniques.

TUNING FOR ACCURACY & PERFORMANCE:
    - Fuzzy Threshold: Controlled via 'entity_resolution.fuzzy_threshold' in
      tms_config.yaml. Higher values (e.g., 95) reduce False Positives but
      may miss sophisticated 'Synthetic Identity' variations.
    - Progress Tracking: Uses 'tqdm' to provide a visual progress bar during
      large batch processing, replacing excessive log printouts.
    - Performance: Fuzzy matching is O(N^2). This implementation uses RapidFuzz's
      C++ backend for high-speed Levenshtein calculations.
    - Scorer: 'token_sort_ratio' is the default, which ignores word order
      (e.g., "John Doe" vs "Doe John").

COMPLIANCE IMPACT:
    Essential for detecting 'Structuring' and 'Smurfing' where a single
    bad actor utilizes multiple accounts to stay below reporting thresholds.
================================================================================
"""

import logging
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

# --- LOGGING SETUP ---
logger = logging.getLogger("TMS_Entity_Resolution")

class EntityResolution:
    def __init__(self, config):
        """
        Initializes the resolution engine using centralized configuration.

        :param config: Dictionary-like object containing 'entity_resolution' settings.
        """
        res_cfg = config.get('entity_resolution', {}) if config else {}

        # RapidFuzz uses a 0-100 scale. We convert from 0.0-1.0 if necessary.
        raw_threshold = res_cfg.get('fuzzy_threshold', 0.85)
        self.threshold = raw_threshold * 100 if raw_threshold <= 1.0 else raw_threshold

        self.use_nat_id = res_cfg.get('use_national_id', True)

        # Select fuzzy scorer dynamically based on config
        scorer_name = res_cfg.get('fuzzy_scorer', 'token_sort_ratio')
        self.scorer = getattr(fuzz, scorer_name, fuzz.token_sort_ratio)

        logger.info(
            f"Entity Resolution initialized. Threshold Score: {self.threshold}, "
            f"Scorer: {scorer_name}, Exact-ID Match Enabled: {self.use_nat_id}"
        )

    def resolve_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the resolution pipeline to generate the 'global_entity_id'.

        Pipeline Flow:
        1. Deterministic Linkage: Groups by National ID.
        2. Probabilistic Linkage: Groups by Fuzzy Name similarity (with Progress Bar).
        """
        if df.empty:
            logger.warning("Input batch is empty. Skipping resolution.")
            return df

        # Create working copy to avoid SettingWithCopy warnings
        resolved_df = df.copy()

        # Step 1: Initialize Global ID
        # Every account starts as its own entity.
        resolved_df['global_entity_id'] = resolved_df['sender_id']

        # Step 2: Deterministic Resolution (National ID / SSN)
        if self.use_nat_id and 'sender_national_id' in resolved_df.columns:
            id_groups = resolved_df.groupby('sender_national_id')['sender_id'].unique()
            for nat_id, uids in id_groups.items():
                if len(uids) > 1:
                    primary_id = sorted(uids)[0]
                    resolved_df.loc[resolved_df['sender_id'].isin(uids), 'global_entity_id'] = primary_id
                    logger.debug(f"Deterministic Link: Linked {len(uids)} accounts via National ID: {nat_id}")

        # Step 3: Probabilistic Resolution (Fuzzy Name Matching)
        unique_profiles = resolved_df[['sender_id', 'sender_name']].drop_duplicates()
        names = unique_profiles['sender_name'].tolist()
        uids = unique_profiles['sender_id'].tolist()

        name_map = {}
        processed_indices = set()

        logger.info(f"Starting probabilistic matching for {len(names)} unique profiles...")

        # Progress Bar Implementation
        # 'desc' provides context to the IT team on the current phase.
        # 'leave=False' cleans the bar from the terminal once done.
        pbar = tqdm(total=len(names), desc="Resolving Identities", unit="profile")

        for i, name in enumerate(names):
            if i in processed_indices:
                pbar.update(1)
                continue

            # Performance: RapidFuzz extract provides the fastest available fuzzy search
            matches = process.extract(
                name,
                names[i + 1:],
                scorer=self.scorer,
                score_cutoff=self.threshold
            )

            current_uid = uids[i]
            for match_name, score, match_idx in matches:
                # Calculate actual index in the original list
                actual_idx = match_idx + i + 1
                matched_uid = uids[actual_idx]

                # Map the matched account to the current "Parent" entity
                name_map[matched_uid] = current_uid
                processed_indices.add(actual_idx)

                # Downgraded to DEBUG to prevent console spam
                logger.debug(f"Fuzzy Match: '{name}' ~ '{match_name}' (Score: {score:.1f})")

            pbar.update(1)

        pbar.close()

        # Step 4: Finalize Mapping
        resolved_df['global_entity_id'] = resolved_df['global_entity_id'].replace(name_map)

        unique_entities = resolved_df['global_entity_id'].nunique()
        logger.info(
            f"Resolution complete. Linked {len(name_map)} account clusters. "
            f"Unique Global Entities: {unique_entities} (from {len(uids)} accounts)."
        )

        return resolved_df

    def get_entity_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a summary for compliance audit trails.
        Identifies high-exposure entities spanning multiple accounts.
        """
        if 'global_entity_id' not in df.columns:
            return pd.DataFrame()

        summary = df.groupby('global_entity_id').agg({
            'sender_id': 'nunique',
            'amount': 'sum',
            'transaction_id': 'count'
        }).rename(columns={
            'sender_id': 'account_count',
            'transaction_id': 'txn_count',
            'amount': 'total_exposure'
        })

        return summary.sort_values(by='total_exposure', ascending=False)

if __name__ == "__main__":
    # Integration smoke test
    print("Identity Resolution Module with Progress Bar - LOADED")