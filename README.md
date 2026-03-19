# Hybrid Machine Learning Transaction Monitoring System (TMS) 

## 2. Executive Summary (The "Why")

### The Problem: Legacy Fragility vs. Sophisticated Typologies
Traditional **Transaction Monitoring Systems (TMS)** are built on rigid, linear "If-Then" logic that struggles to keep pace with modern financial crime. Legacy systems suffer from three primary failures:
* **The Threshold Trap:** Money launderers easily bypass static rules through **Structuring** (Smurfing) and **Micro-structuring**, staying just below reporting limits (e.g., $9,900) to remain invisible to "dumb" filters.
* **False Positive Fatigue:** Linear rules lack context, often flagging legitimate high-volume businesses or seasonal spending spikes, overwhelming **Financial Crime Units (FCU)** with 95%+ noise.
* **The "Unknown Unknown" Blindspot:** Rule-based systems cannot detect new, emerging laundering typologies (e.g., U-Turn integration or synthetic identity clusters) until a regulatory breach has already occurred.

### The Solution: The Quad-Silo Hybrid Architecture
**Global Sentinel** solves the legacy bottleneck by moving away from binary detection to a **Multi-Layered Risk Orchestration** model. Our unique "Quad-Silo" approach aggregates four distinct detection methodologies into a single, normalized 0-100 Risk Score:

1.  **Expert Rule Silo:** High-fidelity deterministic flags for known regulatory breaches (Velocity, Geo-Risk, Sanctions).
2.  **Behavioral Silo:** Dynamic Z-Score ($\mu + \sigma$) profiling that identifies "Out-of-Character" spikes relative to a user's unique 90-day historical baseline.
3.  **Unsupervised ML Silo:** An **Isolation Forest** engine that scans the entire population for statistical outliers—catching the "Unknown Unknowns" that no human-written rule could predict.
4.  **Supervised ML Silo:** A production-optimized **XGBoost** classifier that matches current transaction features against verified historical **SAR (Suspicious Activity Report)** patterns to predict the probability of true illicit intent.

### Regulatory Alignment: Operationalizing the Risk-Based Approach (RBA)
Global Sentinel is architected to exceed the stringent requirements set by **FATF 40 Recommendations**, **FinCEN**, and **6AMLD**. 

By utilizing **Weighted Risk Aggregation**, the system empowers institutions to implement a true **Risk-Based Approach (RBA)**. Instead of a "one-size-fits-all" threshold, the engine automatically prioritizes resources toward **High-Risk and Critical** alerts. This ensures that Compliance Officers focus their investigative capacity on the most credible threats, significantly improving the "Quality-to-Quantity" ratio of regulatory filings and ensuring an audit-ready trail for examiners.

---


## 3. Core Detection Pillars (Technical Deep Dive)

The Global Sentinel engine operates on a **defense-in-depth** model, where four distinct detection silos analyze every transaction concurrently. This ensures that even if a launderer bypasses one layer (e.g., staying under a rule threshold), they are caught by another (e.g., statistical anomaly or network link).

---

### Silo 1: Deterministic Expert Systems (Rules-Based)
This layer handles "Known-Knowns"—clear regulatory breaches and established red flags. It provides the high-fidelity audit trail required by regulators for **CTR (Currency Transaction Reporting)** and **Sanctions Compliance**.
* **Structuring & Smurfing Detection:** Vectorized logic scans for "Just-below-threshold" patterns (e.g., $9,000–$9,999) across 24-hour and 7-day windows.
* **Velocity & Frequency Limits:** Identifies **Rapid Movement of Funds (RMF)** by tracking transaction counts against a configurable sliding window (e.g., >5 txns/hour).

### Silo 2: Behavioral Profiling (Statistical Deviations)
Rather than using "one-size-fits-all" limits, this silo treats every user as their own baseline. It identifies "Out-of-Character" spikes that indicate potential **Account Takeover (ATO)** or sudden laundering activity.
* **Dynamic Z-Score Analysis:** Uses the formula $Z = \frac{x - \mu}{\sigma}$ to flag transactions that deviate significantly from a user's 90-day moving average ($\mu$) and standard deviation ($\sigma$).
* **Temporal Anomaly Detection:** Flags "Unusual Active Windows" (e.g., 1 AM–4 AM) often associated with automated scripts, mule-herding, or international time-zone offsets.
* **Dormancy Awakening:** Specifically monitors accounts that have been inactive for 180+ days and suddenly exhibit high-value throughput.

### Silo 3: Network Intelligence (Graph & Identity)
Financial crime rarely happens in isolation. This silo exposes the "Hidden Web" of relationships between seemingly unrelated accounts to detect **Mule Networks** and **Circular Layering**.
* **Fuzzy Identity Resolution:** Utilizes **RapidFuzz** (`token_sort_ratio`) to merge accounts with slightly altered names or IDs, exposing **Synthetic Identities**.
* **Graph-Link Analytics:** Employs **NetworkX** to calculate **Node Centrality** and **In-Degree/Out-Degree** ratios. This identifies "Mule Hubs" (many-to-one) and "Smurfing Disbursement" (one-to-many) patterns.
* **Cycle Detection:** Scans for circular fund movements (A → B → C → A) used to obfuscate the origin of illicit wealth.

### Silo 4: Machine Learning Overlay (AI Orchestration)
The AI layer provides the final "Risk Boost," identifying complex patterns too subtle for human-written rules.
* **Unsupervised Anomaly (Isolation Forest):** Scans the entire population to find statistical "Islands"—transactions that are anomalous across 15+ dimensions (amount, time, location, device, etc.).
* **Supervised Classification (XGBoost):** A high-performance Gradient Boosting model that acts as a "Pattern Matcher." It compares the current batch against millions of historical features from previous **SAR (Suspicious Activity Report)** filings to predict the likelihood of a true positive.
* **Explainable Scoring:** Every ML output is normalized to a 0-100 scale, allowing investigators to see exactly how much the AI contributed to the final alert.

---

## 4. System Architecture

The Global Sentinel architecture is designed as a **Linear Processing Pipeline** with modular "pluggable" detection silos. This ensures that each stage of the AML lifecycle is isolated, audit-ready, and computationally optimized.

---

### The 7-Stage "Hybrid Detection" Pipeline
Data flows through a high-performance, stateless pipeline that transforms raw transaction logs into prioritized, risk-banded alerts.

1.  **Ingestion & Validation:** Raw extracts are ingested via a **Delta Logic Gate**. The `StateManager` ensures that only new, unprocessed transactions enter the pipeline, preventing duplicate alerts and redundant compute costs.
2.  **Identity Resolution (ER):** Disparate transaction parties are processed through the `EntityResolution` engine. Using **RapidFuzz**, the system links accounts with minor name variations or shared attributes into a single **Global Entity ID**.
3.  **Behavioral Baseline Enrichment:** The system calculates the mean ($\mu$) and standard deviation ($\sigma$) for each entity. Every transaction is then "contextualized" against its own historical baseline to detect deviations.
4.  **Graph & Network Mapping:** Transactions are projected into a directed graph. The `GraphAnalytics` engine calculates **In-Degree (Fan-In)** and **Out-Degree (Fan-Out)** metrics to identify potential money mules and smurfing hubs.
5.  **Multi-Silo Detection:** Every transaction is processed concurrently by the **Rule Engine** (Deterministic), **Behavioral Engine** (Statistical), and **ML Models** (Probabilistic).
6.  **Weighted Risk Orchestration:** The `RiskOrchestrator` aggregates the "Raw Scores" from all silos. Using the weights defined in `tms_config.yaml`, it produces a normalized **Final Risk Score (0-100)**.
7.  **Prioritization & Banding:** Scores are mapped to **Risk Bands** (Low, Medium, High, Critical). High and Critical alerts are flagged for immediate investigator review, while Low/Medium alerts are archived for periodic trend analysis.

---

### Technical Stack & Performance Drivers
Global Sentinel utilizes a "Best-in-Class" stack to balance rapid development with production-grade throughput.

* **Engine Core:** `Python 3.10+` with `Pandas` and `NumPy` for vectorized data manipulation, avoiding slow Python loops.
* **Identity Logic:** `RapidFuzz` (C++ backend) for high-speed string similarity and Levenshtein distance calculations.
* **Network Science:** `NetworkX` for modeling complex transaction topologies and detecting circular layering.
* **AI/ML Layer:** `XGBoost` for gradient-boosted supervised classification and `Scikit-Learn` for unsupervised Isolation Forest anomalies.
* **Persistence:** `Joblib` for efficient serialization of ML artifacts and `YAML` for human-readable governance.

---

### Project Directory Structure
The repository is organized logically by function to allow **Compliance IT** teams to easily locate and tune specific detection modules.

```text
Hybrid-Transaction-Monitoring-System/
├── config/             # GOVERNANCE: YAML thresholds, ML weights, and risk bands.
├── core/               # INFRASTRUCTURE: State Management & Entity Resolution logic.
├── data/               # STORAGE: Raw, Processed, and Model-ready CSV data.
├── engine/             # THE BRAIN: Risk Orchestrator & Behavioral Analytics engine.
├── features/           # ENRICHMENT: Graph Analytics & Network Topology extractors.
├── models/             # AI ARTIFACTS: Serialized XGBoost & Isolation Forest models.
├── rules/              # LOGIC: Deterministic Rules & Complex Typology Scenarios.
├── utils/              # UTILITIES: Unified Logging and Audit Trail formatters.
├── main.py             # ORCHESTRATOR: The primary E2E pipeline entry point.
└── requirements.txt    # ENVIRONMENT: Dependency pinning for deterministic results.
```

---

## 5. Configuration & Governance (The "How")

In a regulated environment, "Black Box" systems are a liability. Global Sentinel is built on the principle of **Transparent Governance**, allowing Compliance Officers and IT Auditors to inspect, tune, and justify every risk score through a centralized configuration layer.

---

### The Single Source of Truth: `tms_config.yaml`
The entire behavior of the engine—from detection sensitivity to ML influence—is controlled by a single, human-readable YAML file. This decoupling of logic from code ensures that:
* **Auditability:** Every change to a threshold (e.g., lowering a structuring limit) is captured in the version control history of the config file.
* **Agility:** Compliance teams can react to new threats (e.g., adding a new High-Risk Jurisdiction) by updating a text file rather than redeploying code.
* **Consistency:** The same thresholds are applied across the Expert Rules, Behavioral Engines, and ML Normalization stages.

### Weighting Logic: The Orchestration Formula
Global Sentinel does not treat all alerts equally. The `RiskOrchestrator` uses a **Weighted Aggregation Model** to blend deterministic "Red Flags" with probabilistic ML scores. 

The final score is derived as:
$$Final\ Risk = (Rule\_Max \times W_1) + (Unsupervised \times W_2) + (Supervised \times W_3)$$

* **Rule Weight ($W_1$):** Typically weighted highest (e.g., 0.45). This ensures that if a clear regulatory breach occurs (like a massive structuring hit), it immediately pushes the transaction into a high-risk band.
* **Anomaly Weight ($W_2$):** Rewards "originality." Even if no rule is broken, a high statistical outlier score from the **Isolation Forest** can elevate a transaction's priority.
* **Supervised Weight ($W_3$):** Acts as a "Pattern Matcher." If a transaction mirrors the features of past SAR filings, this weight boosts the score to ensure investigators see it first.

### Threshold Tuning: Managing the Precision-Recall Tradeoff
A critical challenge in AML is the "False Positive" vs. "Detection Gap" tradeoff. Global Sentinel provides granular control over these levers through **Statistical Thresholding**:

* **Sigma ($\sigma$) & Z-Score Tuning:** In the Behavioral Silo, the `r9_zscore_threshold` (default 3.0) determines how "unusual" a transaction must be to trigger a flag. 
    * **Lowering to 2.0:** Increases "Recall" (catches more subtle spikes) but increases False Positives.
    * **Raising to 4.0:** Increases "Precision" (only flags extreme deviations), reducing the investigator's workload but potentially missing "quiet" launderers.
* **Structuring Floors:** The `r5_structuring_floor` (e.g., $9,000) allows the FCU to define exactly how close to a regulatory limit a customer can get before being flagged, allowing the system to catch "Smurfers" who have studied the bank's reporting limits.

---

### Prerequisites
Before initializing the pipeline, ensure your environment meets the following technical requirements:
* **Runtime:** Python 3.10 or higher.
* **Hardware:** Minimum 8GB RAM (16GB+ recommended for large Graph-Link batch processing).
* **Package Manager:** `pip` (included with Python) or `conda`.

### Step 1: Clone & Environment Setup
Clone the repository and initialize a virtual environment to isolate dependencies and ensure deterministic execution of the ML models.

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

### Step 2: Initialize Configuration
Global Sentinel ships with a `tms_config.yaml` template. Review the **Risk Weights** and **Rule Thresholds** to ensure they align with your institution's specific risk appetite.

```bash
# Verify the configuration directory exists
ls config/tms_config.yaml
```

### Step 3: Execute the E2E Pipeline
You can run the full end-to-end (E2E) pipeline—from data ingestion to risk banding—using the master orchestrator. If no live data is found in `data/raw/`, the system will automatically trigger a synthetic data generator to simulate a production monitoring environment.

```bash
python main.py
```


## 7. Performance & Scalability

In the world of **Transaction Monitoring Systems (TMS)**, performance is not just a technical requirement—it is a regulatory one. A system that cannot process a day's worth of transactions within a 24-hour window creates a "Backlog Risk." Global Sentinel is engineered for high-throughput, utilizing three primary strategies to ensure the engine scales with your institution’s growth.

---

### Vectorization: The Power of Pandas & NumPy
Traditional rule engines often rely on "Row-by-Row" iteration (for-loops), which is computationally expensive in Python. Global Sentinel utilizes **Vectorized Execution**:
* **The Logic:** Operations are applied to entire columns (vectors) of data at once, pushing the heavy lifting down to highly optimized C and Fortran backends.
* **The Impact:** Calculating a **Z-Score** or a **Velocity Flag** across 1 million transactions takes seconds rather than minutes. 
* **Compliance Benefit:** Faster execution allows for more complex "What-If" backtesting, enabling the FCU to simulate new rule thresholds against years of historical data in a single afternoon.

### Delta Processing: Incremental "Watermark" Logic
Processing the same data twice is a waste of compute and a risk for duplicate alerting. The `StateManager` implements a **State Watermark** system:
* **The Logic:** During every run, the system records the `max(timestamp)` or the highest `transaction_id` processed.
* **The Gate:** On the next execution, the `get_delta()` function filters the raw ingestion feed, passing only records newer than the last recorded watermark into the detection silos.
* **Operational Efficiency:** This ensures that whether you run the pipeline every hour or once a day, the compute cost remains strictly proportional to the **new** transaction volume.

### Memory Optimization: Graph Pruning & Batch Controls
Analyzing a "Social Network" of transactions can quickly consume gigabytes of RAM. To prevent system crashes during high-volume spikes, we implement **Surgical Pruning**:
* **Graph Pruning:** The `graph_pruning_threshold` (configured in `tms_config.yaml`) ignores transactions below a certain value (e.g., $1,000) when building the network map. This removes "Micro-Noise" and focuses the engine's memory on high-value potential laundering paths.
* **Batch-Size Controls:** The pipeline supports a `batch_size` parameter. Instead of loading a 10GB CSV into memory, the engine can process data in "Chunks" (e.g., 50,000 records at a time), maintaining a flat memory profile regardless of the total file size.
* **Multiprocessing:** The rule engine can be distributed across multiple CPU cores, allowing the **Deterministic Silo** to run in parallel with the **ML Silo**, effectively halving the total "Time-to-Alert."

---

## 8. Compliance & Auditability

In the highly regulated landscape of **Anti-Money Laundering (AML)**, a detection system is only as good as its ability to withstand a regulatory audit. Global Sentinel is built with "Compliance-First" architecture, ensuring that every automated decision is transparent, reconstructible, and defensible.

---

### The Audit Trail: Structured JSON Logging for SIEM
A common failure in legacy TMS is "Log Fragmentation," where critical detection data is scattered across unformatted text files. Global Sentinel implements **Structured JSON Logging**, specifically designed for seamless integration with Enterprise Security suites like **Splunk**, **ELK Stack (Elasticsearch)**, and **Datadog**.

* **Machine-Readable Context:** Every "Hit" is logged as a structured object containing the `transaction_id`, `entity_id`, the specific `rule_trigger`, and the `raw_score`.
* **Operational Telemetry:** Beyond alerts, the system logs pipeline performance metrics (ingestion time, ML inference latency), providing IT teams with the data needed for **Capacity Planning**.
* **Tamper-Evident Logic:** By piping these logs into a SIEM, institutions maintain an immutable record of when a rule was triggered and how it was weighted, fulfilling the "Full Audit Trail" requirements of **FinCEN** and the **BSA**.

### Explainable AI (XAI): Attribution & Deciphering the "Black Box"
Regulators (and internal Model Risk Management teams) often view Machine Learning with skepticism due to the "Black Box" problem. Global Sentinel solves this through **Risk Attribution**:

* **Silo Attribution:** The final 0-100 score is never presented in isolation. Every alert includes a breakdown of its components (e.g., *“Final Score 85: 40% Velocity Rule, 30% Isolation Forest Anomaly, 15% XGBoost Pattern Match”*).
* **Deterministic Backing:** Even when a high score is driven by ML, the system highlights the underlying **Behavioral Z-Scores** or **Graph Centrality** metrics that influenced the model. This allows a human investigator to "Verify then Trust" the AI's output.
* **Model Lineage:** Every ML inference is tagged with the `model_version` used. This ensures that if a model is updated, investigators can look back at historical alerts and understand the exact logic state of the engine at that specific point in time.

---

## 9\. Licensing & Contact

Global Sentinel is released as an open-source framework to foster collaboration between **RegTech developers** and **Financial Crime Units**. By sharing core detection logic, we aim to standardize the "Hybrid Approach" to AML/CFT monitoring and reduce the global burden of financial crime.

-----

### License

This project is licensed under the **Apache License 2.0**.

* **Permissions:** Commercial use, modification, distribution, and private use are all permitted. This license also includes an explicit grant of patent rights from contributors to users.
* **Conditions:** Users must include a copy of the license and copyright notice in all copies or substantial portions of the software. If you modify the files, you must include prominent notices stating that you changed the files.
* **Disclaimer:** The software is provided "as is," without warranty of any kind. Users are responsible for ensuring their specific implementation and "Derivative Works" meet local regulatory requirements (e.g., FinCEN, FCA).

---

### Contact & Collaboration

We welcome contributions from the community, especially regarding new **Complex Typology Rules** and **ML Model Architectures**.

  * **Lead Architect:** [Rufus Muthusi Mutinda](https://github.com/YourUsername)
  * **Organization:** [PraxiBotics](https://www.google.com/search?q=https://github.com/PraxiBotics) – *AI Automation & Compliance Engineering*
  * **Location:** Nairobi, Kenya 🇰🇪
  * **Inquiries:** For professional consultation, custom rule integration, or enterprise support, please reach out via GitHub Issues or LinkedIn.

-----

### Final Implementation Checklist for GitHub

1.  **README.md:** Copy and paste the expanded sections 1–9 into a single file.
2.  **LICENSE:** Create a file named `LICENSE` in the root directory and paste the MIT License text.
