import csv
import random
import logging
import uuid
import numpy as np
import os
from datetime import datetime, timedelta
from faker import Faker
from tqdm import tqdm

# --- CONFIGURATION & LOGGING ---
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("TMS_Data_Gen_V2_USD")

# Parameters
NUM_TRANSACTIONS = 10_000_000
NUM_CUSTOMERS = 215_000
OUTPUT_FILE = "data/raw/normalized_model.csv"
START_DATE = datetime(2026, 1, 1)
CURRENCY = "USD"
REPORTING_THRESHOLD = 10000.00  # Standard AML threshold for USD

fake = Faker()


def generate_tms_data():
    logger.info(f"Starting generation of Enhanced STR-Ready TMS Data in {CURRENCY}...")

    if not os.path.exists('data/raw'):
        os.makedirs('data/raw', exist_ok=True)

    # --- 1. PRE-GENERATE ENHANCED CUSTOMER PROFILES ---
    logger.info(f"Phase 1/2: Generating {NUM_CUSTOMERS} User Profiles...")
    customer_db = {}
    peer_groups = ["STUDENT", "RETAIL", "SME", "CORPORATE", "HNI"]
    countries = ["US", "CA", "GB", "DE", "SG", "KY", "PA", "CH", "AE", "HK"]
    occupations = ["Software Engineer", "Business Owner", "Unemployed", "Student", "Chief Executive", "Consultant",
                   "Art Dealer"]

    for _ in tqdm(range(NUM_CUSTOMERS), desc="Creating KYC/STR Master List"):
        u_id = f"USR_{uuid.uuid4().hex[:8].upper()}"
        group = random.choice(peer_groups)

        stats = {
            "STUDENT": (40, 25),
            "RETAIL": (150, 90),
            "SME": (2500, 1200),
            "CORPORATE": (45000, 20000),
            "HNI": (8000, 3500)
        }
        avg, std = stats[group]

        first_name = fake.first_name()
        last_name = fake.last_name()
        customer_db[u_id] = {
            "user_id": u_id,
            "full_name": f"{first_name} {last_name}",
            "national_id": fake.ssn() if random.random() > 0.2 else fake.bothify(text='#########'),
            "contact_id": fake.email(),
            "phone_number": fake.phone_number(),
            "occupation": random.choice(occupations),
            "physical_address": fake.address().replace('\n', ', '),
            "account_age_days": random.randint(1, 3650),
            "last_active_date": (START_DATE - timedelta(days=random.randint(0, 400))).strftime('%Y-%m-%d'),
            "user_risk_score": random.randint(1, 100),
            "peer_group": group,
            "expected_monthly_vol": avg * 8,
            "historical_avg_txn": avg,
            "historical_std_dev": std,
            "known_counterparties_count": random.randint(0, 50),
            "primary_device": f"DVC-{uuid.uuid4().hex[:6].upper()}",
            "primary_ip": fake.ipv4()
        }

    # --- 2. GENERATE TRANSACTION LOGS ---
    logger.info(f"Phase 2/2: Generating {NUM_TRANSACTIONS} Transactions...")

    fieldnames = [
        "transaction_id", "timestamp", "sender_id", "receiver_id", "amount", "currency",
        "device_id", "ip_address", "lat_long", "receiver_bank_code", "receiver_country", "channel",
        "sender_name", "sender_contact_id", "sender_phone",
        "receiver_name", "receiver_contact_id",
        "transaction_type", "mcc_code", "transaction_status", "receiver_risk_score",
        "sender_national_id", "sender_occupation", "sender_address",
        "user_id", "account_age_days", "user_risk_score", "peer_group",
        "expected_monthly_vol", "historical_avg_txn", "historical_std_dev", "known_counterparties",
        "is_sar_filed"  # NEW: Supervised Learning Label
    ]

    txn_types = ["P2P", "CASH_OUT", "MERCHANT_PAY", "ASSET_PURCHASE", "WIRE_TRANSFER"]
    mcc_codes = [5411, 5812, 5944, 6011, 6211, 7995]

    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        customer_ids = list(customer_db.keys())

        for i in tqdm(range(NUM_TRANSACTIONS), desc="Synthesizing Transactions"):
            s_id = random.choice(customer_ids)
            profile = customer_db[s_id]

            if i % 10 == 0:
                r_id = customer_ids[0]  # Mule hub
            else:
                r_id = random.choice(customer_ids)

            while r_id == s_id:
                r_id = random.choice(customer_ids)

            r_profile = customer_db[r_id]
            amount = max(1.50, round(np.random.normal(profile["historical_avg_txn"], profile["historical_std_dev"]), 2))

            # Labeling initialization
            is_sar_filed = 0

            # --- OUTLIER INJECTION & LABELING LOGIC ---

            # 1. Structuring Outliers (Rule 5: Just below $10k threshold)
            if i % 120 == 0:
                amount = round(random.uniform(9800.00, 9999.50), 2)
                t_type = "WIRE_TRANSFER"
                # Labeling: High-probability structuring is marked as SAR
                if random.random() > 0.3:
                    is_sar_filed = 1

            # 2. Velocity/High Value Outliers
            elif i % 70 == 0:
                amount = round(random.uniform(50000.00, 150000.00), 2)
                t_type = "ASSET_PURCHASE"
                # Labeling: New accounts doing massive asset purchases are marked as SAR
                if profile["account_age_days"] < 30:
                    is_sar_filed = 1

            # 3. Standard Randomness
            else:
                t_type = random.choice(txn_types)

            mcc = random.choice(mcc_codes)
            if t_type == "ASSET_PURCHASE":
                mcc = 5944

            writer.writerow({
                "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                "timestamp": (START_DATE + timedelta(seconds=i * random.randint(5, 45))).isoformat(),
                "sender_id": s_id, "receiver_id": r_id, "amount": amount, "currency": CURRENCY,
                "device_id": profile[
                    "primary_device"] if random.random() > 0.05 else f"DVC-{uuid.uuid4().hex[:6].upper()}",
                "ip_address": profile["primary_ip"] if random.random() > 0.1 else fake.ipv4(),
                "lat_long": f"{fake.latitude()},{fake.longitude()}",
                "receiver_bank_code": f"BK-{random.choice(['CHAS', 'WFGO', 'BOFA', 'CITI', 'GSAC'])}",
                "receiver_country": random.choice(countries),
                "channel": random.choice(["MOBILE", "WEB", "WIRE", "ATM"]),
                "sender_name": profile["full_name"], "sender_contact_id": profile["contact_id"],
                "sender_phone": profile["phone_number"],
                "receiver_name": r_profile["full_name"], "receiver_contact_id": r_profile["contact_id"],
                "transaction_type": t_type, "mcc_code": mcc,
                "transaction_status": "COMPLETED" if random.random() > 0.03 else "FAILED",
                "receiver_risk_score": r_profile["user_risk_score"], "sender_national_id": profile["national_id"],
                "sender_occupation": profile["occupation"], "sender_address": profile["physical_address"],
                "user_id": profile["user_id"], "account_age_days": profile["account_age_days"],
                "user_risk_score": profile["user_risk_score"], "peer_group": profile["peer_group"],
                "expected_monthly_vol": profile["expected_monthly_vol"],
                "historical_avg_txn": profile["historical_avg_txn"],
                "historical_std_dev": profile["historical_std_dev"],
                "known_counterparties": profile["known_counterparties_count"],
                "is_sar_filed": is_sar_filed
            })

    logger.info(f"Generation Complete! File saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_tms_data()