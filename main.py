"""
================================================================================
MODULE: TMS HYBRID PIPELINE ORCHESTRATOR (MAIN ENTRY POINT)
================================================================================

DESCRIPTION:
    This is the master orchestration script for the Transaction Monitoring
    System (TMS). It executes a multi-stage "Hybrid Detection" pipeline that
    bridges deterministic logic with advanced behavioral and ML models.

PIPELINE STAGES:
    1. Configuration: Imports YAML parameters for global threshold consistency.
    2. Bootstrap: Ensures Supervised ML artifacts (XGBoost) are trained/available.
    3. Ingestion: Loads raw data and applies "Delta Logic" (processing only new txns).
    4. Identity: Resolves fuzzy sender/receiver data into a 'Global Entity ID'.
    5. Feature Engineering: Extracts Graph Centrality and Temporal Profiles.
    6. Multi-Layer Detection: Runs simple rules, complex typologies, and ML.
    7. Risk Orchestration: Normalizes and weights all signals into Risk Bands.

OPERATIONAL TUNING:
    - Delta Check: Prevents redundant processing. Controlled via StateManager.
    - Rule Aggregation: Now dynamically identifies '_raw' score columns to
      ensure zero-loss handover to the Risk Orchestrator.
    - Output: Generates an audit-ready CSV in /data/processed with full
      attribution of risk scores.

COMPLIANCE IMPACT:
    Ensures a "Single View of Risk" for investigators. By aggregating silos,
    the system identifies complex layering patterns that would pass through
    traditional linear rules undetected.
================================================================================
"""

import os
import yaml
import logging
import pandas as pd
from datetime import datetime

# --- 1. SUBSYSTEM IMPORTS ---
# Logic: Each module handles a specific compliance domain or technical utility.
from utils.logger import setup_logger
from generate_tms_data import generate_tms_data
from core.state_manager import StateManager
from core.entity_resolution import EntityResolution
from engine.risk_orchestrator import RiskOrchestrator
from rules.transaction_rules import TransactionRules
from rules.complex_typologies import ComplexTypologies
from features.behavioral_profiles import BehavioralProfiles
from features.graph_analytics import GraphAnalytics
from models.anomaly_detection import AnomalyDetector
from models.supervised_classifier import SupervisedClassifier

# Initialize Global Unified Logger (Configured for both Console and File output)
logger = setup_logger("TMS_Main")

def load_global_config(config_path="config/tms_config.yaml"):
    """Loads central parameters. Critical for ensuring rule thresholds match across modules."""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        logger.warning(f"Config not found at {config_path}. Using internal defaults.")
        return {}
    except Exception as e:
        logger.error(f"Config Load Error: {e}")
        return {}

def run_pipeline():
    """
    The E2E Execution Loop. Orchestrates the flow of data through detection silos.
    """
    logger.info("==========================================================")
    logger.info("STAGING: Initializing TMS Advanced Hybrid Engine...")
    logger.info("==========================================================")
    start_time = datetime.now()

    # --- 2. CONFIGURATION & CORE SETUP ---
    config = load_global_config()

    # Core Infrastructure
    state_mgr = StateManager(config=config)
    entity_res = EntityResolution(config=config)
    orchestrator = RiskOrchestrator(config=config)

    # Detection & Feature Engines
    txn_rules = TransactionRules(config=config)
    complex_rules = ComplexTypologies(config=config)
    behavioral = BehavioralProfiles(config=config)
    graph_eng = GraphAnalytics(config=config)

    # ML Inference Modules
    unsupervised = AnomalyDetector(config=config)
    supervised = SupervisedClassifier(config=config)

    # Paths
    TRAIN_DATA_PATH = "data/raw/normalized_model.csv"
    LIVE_DATA_PATH = "data/raw/normalized_tms_extract.csv"

    # --- 3. MODEL INTEGRITY CHECK (BOOTSTRAP) ---
    # Ensures the Supervised ML model is ready. If artifact is missing, it triggers training.
    if not os.path.exists(supervised.model_path):
        logger.warning("ML Model artifact missing. Bootstrapping from historical data...")
        if os.path.exists(TRAIN_DATA_PATH):
            train_df = pd.read_csv(TRAIN_DATA_PATH)
            supervised.train_model(train_df, label_col='is_sar_filed')
        else:
            logger.error("Bootstrap failed: No training data. Supervised scoring will be 0.0.")

    # --- 4. DATA INGESTION & DELTA GATE ---
    if not os.path.exists(LIVE_DATA_PATH):
        logger.info("Target extract missing. Invoking Synthetic Generator for simulation...")
        generate_tms_data()

    full_df = pd.read_csv(LIVE_DATA_PATH)

    # Delta logic ensures we only spend compute on 'New' transactions since the last run.
    df = state_mgr.get_delta(full_df)

    if df.empty:
        logger.info("DELTA CHECK: No new transactions. Pipeline hibernation mode.")
        return

    # --- 5. PHASE 1: IDENTITY & GRAPH ENRICHMENT ---
    logger.info(f"PHASE 1: Resolving {len(df)} records into entity networks...")

    # Resolve 'Global Entity IDs' to link disparate accounts to single humans
    df = entity_res.resolve_entities(df)

    # Build the 'Graph' (Social Network) of the batch to find Centrality/Risk-by-Association
    graph_features = graph_eng.extract_graph_features(df)
    df = df.merge(graph_features, on='global_entity_id', how='left')

    # --- 6. PHASE 2: MULTI-SILO DETECTION & SCORING ---
    logger.info("PHASE 2: Executing Multi-Silo Detection Engines...")

    # A. Level 1: Deterministic Rules (Structuring, Velocity, Rapid-Movement)
    # Output: 'velocity_raw', 'structuring_raw', etc.
    df = txn_rules.run_all_rules(df)

    # B. Level 2: Complex Typology Scenarios (Smurfing Hubs, U-Turns)
    # Output: Updates existing '_raw' columns or adds scenario scores.
    df = complex_rules.apply_complex_rules(df)

    # C. Level 3: Behavioral Profiling (Deviation from μ + σ)
    # This identifies "Out-of-Character" spikes for specific entities.
    # Note: Orchestrator handles the actual calculation; we just prep the pipeline here.

    # D. Level 4: Machine Learning Scoring
    df = unsupervised.predict_anomaly_score(df) # Statistical Outliers (Isolation Forest)
    df = supervised.predict_probability(df)    # Probability of SAR (XGBoost)

    # --- 7. PHASE 3: FINAL RISK ORCHESTRATION ---
    logger.info("PHASE 3: Orchestrating Hybrid Risk Scores & Prioritization...")

    # This aggregates all '_raw' scores and ML scores into 'final_risk_score' [0-100]
    processed_df = orchestrator.calculate_final_risk(df)

    # Filter for the 'Compliance Alert Queue'
    alerts = processed_df[processed_df['risk_band'].isin(['HIGH', 'CRITICAL'])]

    # --- 8. PERSISTENCE & CLOSURE ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    save_path = f"{output_dir}/tms_batch_results_{timestamp}.csv"
    processed_df.to_csv(save_path, index=False)

    # Watermark: Update state so we don't re-process these in the next run
    state_mgr.save_state(processed_df)

    # PERFORMANCE SUMMARY for IT Logs
    duration = datetime.now() - start_time
    logger.info("==========================================================")
    logger.info(f"BATCH SUCCESS: {len(processed_df)} txns processed.")
    logger.info(f"ALERTS GENERATED: {len(alerts)} High/Critical.")
    logger.info(f"TOTAL RUNTIME: {duration}")
    logger.info(f"AUDIT TRAIL SAVED: {save_path}")
    logger.info("==========================================================")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.exception(f"CRITICAL PIPELINE FAILURE: {e}")