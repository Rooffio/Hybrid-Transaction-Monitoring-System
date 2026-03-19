# Hybrid Machine Learning Transaction Monitoring System (TMS)

> **Global Sentinel** — Enterprise-Grade AML/CFT Detection Engine | Risk-Based Approach (RBA) | FATF 40 Recommendations Compliant

---

## 📋 Table of Contents

- [Executive Summary](#2-executive-summary-the-why)
- [Core Detection Pillars](#3-core-detection-pillars-technical-deep-dive)
- [System Architecture](#4-system-architecture)
- [Configuration & Governance](#5-configuration--governance-the-how)
- [Installation & Deployment](#6-installation--deployment)
- [Performance & Scalability](#7-performance--scalability)
- [Compliance & Auditability](#8-compliance--auditability)
- [Execution Logs](#-tms-advanced-hybrid-engine--execution-log)
- [Licensing & Contact](#9-licensing--contact)

---

## 2. Executive Summary (The "Why")

### ⚠️ The Problem: Legacy Fragility vs. Sophisticated Typologies

Traditional **Transaction Monitoring Systems (TMS)** are built on rigid, linear *"If-Then"* logic that struggles to keep pace with modern financial crime. Legacy systems suffer from **three primary failures**:

| Failure Mode | Description | Impact |
|--------------|-------------|--------|
| **The Threshold Trap** | Money launderers bypass static rules through **Structuring** (Smurfing) and **Micro-structuring**, staying just below reporting limits (e.g., $9,900) | Invisible to "dumb" filters |
| **False Positive Fatigue** | Linear rules lack context, flagging legitimate high-volume businesses or seasonal spending spikes | **95%+ noise** overwhelming Financial Crime Units (FCU) |
| **The "Unknown Unknown" Blindspot** | Rule-based systems cannot detect new, emerging laundering typologies until a regulatory breach has occurred | Reactive vs. Proactive detection |

---

### ✅ The Solution: The Quad-Silo Hybrid Architecture

**Global Sentinel** solves the legacy bottleneck by moving away from binary detection to a **Multi-Layered Risk Orchestration** model. Our unique **"Quad-Silo"** approach aggregates four distinct detection methodologies into a single, normalized **0–100 Risk Score**:

| Silo | Detection Method | Use Case |
|------|------------------|----------|
| **1. Expert Rule Silo** | High-fidelity deterministic flags | Known regulatory breaches (Velocity, Geo-Risk, Sanctions) |
| **2. Behavioral Silo** | Dynamic Z-Score ($\mu + \sigma$) profiling | "Out-of-Character" spikes vs. 90-day historical baseline |
| **3. Unsupervised ML Silo** | **Isolation Forest** engine | Statistical outliers — catches "Unknown Unknowns" |
| **4. Supervised ML Silo** | Production-optimized **XGBoost** classifier | Matches transaction features against verified **SAR** patterns |

---

### 🏛️ Regulatory Alignment: Operationalizing the Risk-Based Approach (RBA)

Global Sentinel is architected to exceed stringent requirements set by:

- **FATF 40 Recommendations**
- **FinCEN** (Financial Crimes Enforcement Network)
- **6AMLD** (6th Anti-Money Laundering Directive)

By utilizing **Weighted Risk Aggregation**, the system empowers institutions to implement a true **Risk-Based Approach (RBA)**. Instead of a *"one-size-fits-all"* threshold, the engine automatically prioritizes resources toward **High-Risk** and **Critical** alerts.

**Key Benefits:**
- ✅ Compliance Officers focus investigative capacity on the most credible threats
- ✅ Significant improvement in **"Quality-to-Quantity"** ratio of regulatory filings
- ✅ Audit-ready trail for regulatory examiners

---

## 3. Core Detection Pillars (Technical Deep Dive)

The Global Sentinel engine operates on a **defense-in-depth** model, where four distinct detection silos analyze every transaction **concurrently**. This ensures that even if a launderer bypasses one layer, they are caught by another.

---

### 🎯 Silo 1: Deterministic Expert Systems (Rules-Based)

This layer handles **"Known-Knowns"** — clear regulatory breaches and established red flags. It provides the high-fidelity audit trail required by regulators for **CTR (Currency Transaction Reporting)** and **Sanctions Compliance**.

| Detection Type | Logic | Window |
|----------------|-------|--------|
| **Structuring & Smurfing** | Vectorized logic scans for "Just-below-threshold" patterns (e.g., $9,000–$9,999) | 24-hour / 7-day |
| **Velocity & Frequency** | Tracks transaction counts against configurable sliding window | e.g., >5 txns/hour |
| **Rapid Movement of Funds (RMF)** | Identifies high-velocity fund transfers | Configurable |

---

### 📈 Silo 2: Behavioral Profiling (Statistical Deviations)

Rather than using *"one-size-fits-all"* limits, this silo treats **every user as their own baseline**. It identifies *"Out-of-Character"* spikes that indicate potential **Account Takeover (ATO)** or sudden laundering activity.

**Core Formulas:**

$$Z = \frac{x - \mu}{\sigma}$$

Where:
- $x$ = Current transaction value
- $\mu$ = 90-day moving average
- $\sigma$ = Standard deviation

| Anomaly Type | Detection Logic | Risk Indicator |
|--------------|-----------------|----------------|
| **Dynamic Z-Score Analysis** | Flags transactions deviating from 90-day baseline | Sudden laundering activity |
| **Temporal Anomaly Detection** | Flags "Unusual Active Windows" (e.g., 1 AM–4 AM) | Automated scripts, mule-herding |
| **Dormancy Awakening** | Monitors accounts inactive 180+ days with sudden high-value throughput | Account compromise |

---

### 🔗 Silo 3: Network Intelligence (Graph & Identity)

Financial crime rarely happens in isolation. This silo exposes the **"Hidden Web"** of relationships between seemingly unrelated accounts to detect **Mule Networks** and **Circular Layering**.

| Capability | Technology | Detection Target |
|------------|------------|------------------|
| **Fuzzy Identity Resolution** | `RapidFuzz` (`token_sort_ratio`) | **Synthetic Identities** |
| **Graph-Link Analytics** | `NetworkX` — Node Centrality, In-Degree/Out-Degree | **Mule Hubs** (many-to-one), **Smurfing Disbursement** (one-to-many) |
| **Cycle Detection** | Circular fund movement scanning (A → B → C → A) | Obfuscated illicit wealth origin |

---

### 🤖 Silo 4: Machine Learning Overlay (AI Orchestration)

The AI layer provides the final **"Risk Boost"**, identifying complex patterns too subtle for human-written rules.

| Model Type | Algorithm | Function |
|------------|-----------|----------|
| **Unsupervised Anomaly** | Isolation Forest | Scans entire population for statistical "Islands" across 15+ dimensions |
| **Supervised Classification** | XGBoost (Gradient Boosting) | "Pattern Matcher" against historical **SAR** filings |
| **Explainable Scoring** | Normalized 0–100 scale | Shows AI contribution to final alert |

---

## 4. System Architecture

The Global Sentinel architecture is designed as a **Linear Processing Pipeline** with modular *"pluggable"* detection silos. Each stage of the AML lifecycle is isolated, audit-ready, and computationally optimized.

---

### 🔄 The 7-Stage "Hybrid Detection" Pipeline

Data flows through a high-performance, stateless pipeline that transforms raw transaction logs into prioritized, risk-banded alerts.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           GLOBAL SENTINEL PIPELINE                              │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────────────────┤
│   1.    │   2.    │   3.    │   4.    │   5.    │   6.    │        7.           │
│Ingestion│Identity │Behavior │  Graph  │  Multi  │  Risk   │  Prioritization     │
│    &    │Resolution│Baseline │   &     │  Silo   │Orchestr-│      & Banding     │
│Validation│  (ER)   │Enrich.  │ Network │Detection|  ator   │                     │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────────────────┘
```

| Stage | Component | Function |
|-------|-----------|----------|
| **1. Ingestion & Validation** | `Delta Logic Gate`, `StateManager` | Only new, unprocessed transactions enter pipeline |
| **2. Identity Resolution (ER)** | `EntityResolution`, `RapidFuzz` | Links accounts with minor variations into **Global Entity ID** |
| **3. Behavioral Baseline Enrichment** | $\mu$, $\sigma$ calculation | Contextualizes transactions against historical baseline |
| **4. Graph & Network Mapping** | `GraphAnalytics` | Calculates **In-Degree (Fan-In)** and **Out-Degree (Fan-Out)** |
| **5. Multi-Silo Detection** | Rule + Behavioral + ML Engines | Concurrent processing across all detection layers |
| **6. Weighted Risk Orchestration** | `RiskOrchestrator` | Aggregates raw scores into **Final Risk Score (0–100)** |
| **7. Prioritization & Banding** | Risk Bands (Low/Medium/High/Critical) | Flags High/Critical for immediate review |

---

### 🛠️ Technical Stack & Performance Drivers

Global Sentinel utilizes a **"Best-in-Class"** stack to balance rapid development with production-grade throughput.

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Engine Core** | `Python 3.10+`, `Pandas`, `NumPy` | Vectorized data manipulation |
| **Identity Logic** | `RapidFuzz` (C++ backend) | High-speed string similarity, Levenshtein distance |
| **Network Science** | `NetworkX` | Transaction topology modeling, circular layering detection |
| **AI/ML Layer** | `XGBoost`, `Scikit-Learn` | Gradient-boosted classification, Isolation Forest anomalies |
| **Persistence** | `Joblib`, `YAML` | ML artifact serialization, human-readable governance |

---

### 📁 Project Directory Structure

```
Hybrid-Transaction-Monitoring-System/
├── config/             # GOVERNANCE: YAML thresholds, ML weights, risk bands
├── core/               # INFRASTRUCTURE: State Management & Entity Resolution
├── data/               # STORAGE: Raw, Processed, Model-ready CSV data
├── engine/             # THE BRAIN: Risk Orchestrator & Behavioral Analytics
├── features/           # ENRICHMENT: Graph Analytics & Network Topology
├── models/             # AI ARTIFACTS: Serialized XGBoost & Isolation Forest
├── rules/              # LOGIC: Deterministic Rules & Complex Typologies
├── utils/              # UTILITIES: Unified Logging & Audit Trail formatters
├── main.py             # ORCHESTRATOR: Primary E2E pipeline entry point
└── requirements.txt    # ENVIRONMENT: Dependency pinning for deterministic results
```

---

## 5. Configuration & Governance (The "How")

In a regulated environment, **"Black Box"** systems are a liability. Global Sentinel is built on the principle of **Transparent Governance**, allowing Compliance Officers and IT Auditors to inspect, tune, and justify every risk score through a centralized configuration layer.

---

### 📄 The Single Source of Truth: `tms_config.yaml`

The entire behavior of the engine — from detection sensitivity to ML influence — is controlled by a single, human-readable **YAML** file.

| Benefit | Description |
|---------|-------------|
| **Auditability** | Every threshold change captured in version control history |
| **Agility** | Compliance teams react to new threats by updating text, not redeploying code |
| **Consistency** | Same thresholds applied across Expert Rules, Behavioral Engines, and ML Normalization |

---

### ⚖️ Weighting Logic: The Orchestration Formula

Global Sentinel does not treat all alerts equally. The `RiskOrchestrator` uses a **Weighted Aggregation Model** to blend deterministic *"Red Flags"* with probabilistic ML scores.

**Final Score Derivation:**

$$Final\ Risk = (Rule\_Max \times W_1) + (Unsupervised \times W_2) + (Supervised \times W_3)$$

| Weight Component | Default Value | Purpose |
|------------------|---------------|---------|
| **Rule Weight ($W_1$)** | 0.45 | Clear regulatory breaches push transaction to high-risk band immediately |
| **Anomaly Weight ($W_2$)** | 0.30 | High statistical outlier score from **Isolation Forest** elevates priority |
| **Supervised Weight ($W_3$)** | 0.25 | **Pattern Matcher** — mirrors features of past **SAR** filings |

---

### 🎚️ Threshold Tuning: Managing the Precision-Recall Tradeoff

| Parameter | Default | Lower Value Effect | Higher Value Effect |
|-----------|---------|-------------------|---------------------|
| **`r9_zscore_threshold`** | 3.0σ | ↑ Recall (catches subtle spikes), ↑ False Positives | ↑ Precision (extreme deviations only), ↓ Workload |
| **`r5_structuring_floor`** | $9,000 | Catches more "Smurfers" near limits | Reduces noise, may miss quiet launderers |

---

## 6. Installation & Deployment

### 📦 Prerequisites

| Requirement | Specification |
|-------------|---------------|
| **Runtime** | Python 3.10 or higher |
| **Hardware** | Minimum 8GB RAM (16GB+ recommended for large Graph-Link batch processing) |
| **Package Manager** | `pip` (included with Python) or `conda` |

---

### Step 1: Clone & Environment Setup

```bash
# Clone the repository
git clone https://github.com/YourUsername/Hybrid-Transaction-Monitoring-System.git
cd Hybrid-Transaction-Monitoring-System

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install production dependencies
pip install -r requirements.txt
```

---

### Step 2: Initialize Configuration

```bash
# Verify the configuration directory exists
ls config/tms_config.yaml
```

> **Note:** Review **Risk Weights** and **Rule Thresholds** to ensure alignment with your institution's specific risk appetite.

---

### Step 3: Execute the E2E Pipeline

```bash
python main.py
```

> **Behavior:** If no live data is found in `data/raw/`, the system automatically triggers a synthetic data generator to simulate a production monitoring environment.

---

## 7. Performance & Scalability

In the world of **Transaction Monitoring Systems (TMS)**, performance is not just a technical requirement — it is a **regulatory one**. A system that cannot process a day's worth of transactions within a 24-hour window creates a **"Backlog Risk."**

---

### 🚀 Vectorization: The Power of Pandas & NumPy

| Aspect | Traditional Approach | Global Sentinel Approach |
|--------|---------------------|-------------------------|
| **Execution Model** | Row-by-Row iteration (for-loops) | **Vectorized Execution** (entire columns at once) |
| **Backend** | Python interpreter | Optimized C and Fortran backends |
| **Impact** | Minutes for 1M transactions | **Seconds** for 1M transactions |

**Compliance Benefit:** Faster execution enables complex *"What-If"* backtesting — FCU can simulate new rule thresholds against years of historical data in a single afternoon.

---

### 💧 Delta Processing: Incremental "Watermark" Logic

| Component | Function |
|-----------|----------|
| **State Watermark** | Records `max(timestamp)` or highest `transaction_id` processed per run |
| **`get_delta()` Gate** | Filters raw ingestion feed, passing only records newer than last watermark |
| **Operational Efficiency** | Compute cost proportional to **new** transaction volume only |

---

### 🧠 Memory Optimization: Graph Pruning & Batch Controls

| Strategy | Configuration | Benefit |
|----------|---------------|---------|
| **Graph Pruning** | `graph_pruning_threshold` (e.g., $1,000) | Removes "Micro-Noise", focuses on high-value laundering paths |
| **Batch-Size Controls** | `batch_size` parameter (e.g., 50,000 records/chunk) | Flat memory profile regardless of total file size |
| **Multiprocessing** | Distributed across CPU cores | **Deterministic Silo** runs parallel with **ML Silo** — halves "Time-to-Alert" |

---

## 8. Compliance & Auditability

In the highly regulated landscape of **Anti-Money Laundering (AML)**, a detection system is only as good as its ability to withstand a **regulatory audit**.

---

### 📝 The Audit Trail: Structured JSON Logging for SIEM

| Feature | Description | Integration Target |
|---------|-------------|-------------------|
| **Machine-Readable Context** | Every "Hit" logged as structured object (`transaction_id`, `entity_id`, `rule_trigger`, `raw_score`) | **Splunk**, **ELK Stack**, **Datadog** |
| **Operational Telemetry** | Pipeline performance metrics (ingestion time, ML inference latency) | Capacity Planning |
| **Tamper-Evident Logic** | Immutable record of rule triggers and weighting | **FinCEN**, **BSA** compliance |

---

### 🔍 Explainable AI (XAI): Attribution & Deciphering the "Black Box"

| Capability | Implementation | Regulatory Benefit |
|------------|----------------|-------------------|
| **Silo Attribution** | Final score breakdown (e.g., *"85: 40% Velocity Rule, 30% Isolation Forest, 15% XGBoost"*) | Transparent decision-making |
| **Deterministic Backing** | Highlights underlying **Behavioral Z-Scores** or **Graph Centrality** metrics | Human investigator can *"Verify then Trust"* |
| **Model Lineage** | Every ML inference tagged with `model_version` | Historical alert reconstruction |

---

## 📊 TMS Advanced Hybrid Engine — Execution Log

### 🚀 System Initialization

```text
2026-03-19 18:19:49,899 [INFO] ==========================================================
2026-03-19 18:19:49,899 [INFO] STAGING: Initializing TMS Advanced Hybrid Engine...
2026-03-19 18:19:49,899 [INFO] ==========================================================
2026-03-19 18:19:49,914 [INFO] StateManager initialized. Monitoring watermark at: data/cache\last_run_metadata.json
2026-03-19 18:19:49,914 [INFO] Entity Resolution initialized. Threshold Score: 85.0, Scorer: token_sort_ratio, Exact-ID Match Enabled: True
2026-03-19 18:19:49,914 [INFO] Behavioral Engine initialized with 3.0σ threshold.
2026-03-19 18:19:49,914 [INFO] Orchestrator Weights: Rules(0.45), Anomaly(0.3), Supervised(0.25)
2026-03-19 18:19:49,914 [INFO] Transaction Rules Engine initialized. Structuring Floor: $9000, Velocity Multiplier: 2.0x, Monitored Jurisdictions: 7
2026-03-19 18:19:49,914 [INFO] Complex Typologies Engine initialized. Fan-In Limit: 10, Fan-Out Limit: 10
2026-03-19 18:19:49,914 [INFO] Behavioral Profiling Engine initialized. Threshold: 3.0σ, Min History: 3
2026-03-19 18:19:49,914 [INFO] Graph Engine initialized. Pruning threshold: >10000, PageRank Limit: 5000 nodes.
2026-03-19 18:19:49,914 [INFO] Anomaly Detector initialized. Mode: Batch-Relative. Contamination Target: 1.0%
2026-03-19 18:19:49,914 [INFO] Supervised Engine initialized. Artifact Target: data/cache/xgboost_sar_model.joblib
```

---

### 📥 Data Ingestion & Entity Resolution

```text
2026-03-19 18:19:51,649 [INFO] Delta identified: 100000 new records found. Previous watermark: 1970-01-01T00:00:00.
2026-03-19 18:19:51,649 [INFO] PHASE 1: Resolving 100000 records into entity networks...
2026-03-19 18:19:53,661 [INFO] Starting probabilistic matching for 22706 unique profiles...
Resolving Identities: 100%|██████████| 22706/22706 [02:52<00:00, 131.46profile/s]
2026-03-19 18:23:53,664 [INFO] Resolution complete. Linked 6334 account clusters. Unique Global Entities: 16372 (from 22706 accounts).
2026-03-19 18:23:53,709 [INFO] Filtered graph universe: 24928 edges (Pruned from 100000 total).
```

---

### 🔗 Graph Analytics

```text
2026-03-19 18:23:53,914 [INFO] Scanning top 1000 nodes for circular 'U-Turn' flows...
2026-03-19 18:23:53,988 [INFO] Graph analytics complete. Nodes analyzed: 16829
```

---

### 🧠 Detection Engines Execution

```text
2026-03-19 18:23:54,028 [INFO] PHASE 2: Executing Multi-Silo Detection Engines...
2026-03-19 18:23:54,028 [INFO] Executing Transaction Rules Suite (Deterministic Pass)...
2026-03-19 18:23:54,038 [INFO] Executing Complex Typology Analysis (Smurfing, U-Turns, Integration)...
2026-03-19 18:23:54,390 [INFO] Analysis complete. Identified: 26 Smurfing Hubs, 0 Wash-Traders, 7073 Integration Events.
2026-03-19 18:23:54,390 [INFO] Analyzing 100000 records for statistical outliers...
2026-03-19 18:23:56,626 [INFO] Anomaly detection complete. Max Batch Score: 55.7
2026-03-19 18:23:56,636 [INFO] Successfully loaded SAR prediction model from disk.
2026-03-19 18:23:56,818 [INFO] Supervised scoring complete. Max Prob: 73.0
```

---

### 📊 Risk Scoring & Aggregation

```text
2026-03-19 18:23:56,820 [INFO] PHASE 3: Orchestrating Hybrid Risk Scores & Prioritization...
2026-03-19 18:23:56,865 [INFO] Generating entity-level baselines on: global_entity_id
2026-03-19 18:23:57,012 [INFO] Aggregating 5 signals: ['network_raw', 'structuring_raw', 'velocity_raw', 'geo_raw', 'behavioral_raw']
2026-03-19 18:23:57,062 [INFO] Risk aggregation complete. Max Risk Band: {'LOW': 68951, 'MEDIUM': 30994, 'HIGH': 55}
```

---

### ✅ Execution Summary

```text
2026-03-19 18:24:00,899 [INFO] System state persisted. New Watermark: 2026-02-22T00:36:45
2026-03-19 18:24:00,900 [INFO] ==========================================================
2026-03-19 18:24:00,900 [INFO] BATCH SUCCESS: 100000 txns processed.
2026-03-19 18:24:00,900 [INFO] ALERTS GENERATED: 55 High/Critical.
2026-03-19 18:24:00,900 [INFO] TOTAL RUNTIME: 0:04:11.000006
2026-03-19 18:24:00,900 [INFO] AUDIT TRAIL SAVED: data/processed/tms_batch_results_20260319_1823.csv
2026-03-19 18:24:00,900 [INFO] ==========================================================
```

---

## 9. Licensing & Contact

Global Sentinel is released as an **open-source framework** to foster collaboration between **RegTech developers** and **Financial Crime Units**. By sharing core detection logic, we aim to standardize the *"Hybrid Approach"* to AML/CFT monitoring and reduce the global burden of financial crime.

---

### 📜 License

**Apache License 2.0**

| Permission | Status |
|------------|--------|
| Commercial Use | ✅ Permitted |
| Modification | ✅ Permitted |
| Distribution | ✅ Permitted |
| Private Use | ✅ Permitted |
| Patent Rights | ✅ Explicit grant from contributors |

**Conditions:**
- Include copy of license and copyright notice in all copies
- Modified files must include prominent notices stating changes
- Software provided *"as is"* without warranty

> **Disclaimer:** Users are responsible for ensuring their specific implementation and *"Derivative Works"* meet local regulatory requirements (e.g., **FinCEN**, **FCA**).

---

### 📬 Contact & Collaboration

We welcome contributions from the community, especially regarding new **Complex Typology Rules** and **ML Model Architectures**.

| Role | Details |
|------|---------|
| **Lead Architect** | [Rufus Mutinda](https://github.com/Rooffio) |
| **Organization** | [PraxiBotics](https://www.praxibotics.lovable.app) — *AI Automation & Compliance Engineering* |
| **Inquiries** | GitHub Issues or LinkedIn for professional consultation, custom rule integration, or enterprise support |


---

> **Built for Compliance. Engineered for Scale. Designed for Transparency.**

© 2026 **PraxiBotics** | AI Automation & Compliance Engineering
