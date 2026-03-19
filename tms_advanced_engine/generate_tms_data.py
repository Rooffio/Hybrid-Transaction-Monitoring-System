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
logger = logging.getLogger("TMS_Data_Gen")

# Parameters
NUM_TRANSACTIONS = 100_000
NUM_CUSTOMERS = 15_000
OUTPUT_FILE = "data/enhanced_tms_extract.csv"
START_DATE = datetime(2026, 1, 1)
CURRENCY = "USD"

fake = Faker()


def generate_tms_data():
    """
    Generates a synthetic dataset of bank transactions with
    injected AML typologies (Structuring, Layering, and Mule Hubs).
    """
    logger.info(f"Starting generation of {NUM_TRANSACTIONS} transactions...")

    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    # --- 1. GENERATE CUSTOMER PROFILES ---
    logger.info(f"Phase 1/2: Generating {NUM_CUSTOMERS} User Profiles...")
    customer_db = {}
    peer_groups = ["STUDENT", "RETAIL", "SME", "CORPORATE", "HNI"]
    countries = ["US", "CA", "GB", "DE", "SG", "KY", "PA", "CH", "AE", "HK"]
    occupations = ["Software Engineer", "Business Owner", "Unemployed", "Student", "Chief Executive", "Consultant",
                   "Art Dealer"]

    for _ in range(NUM_CUSTOMERS):
        u_id = f"USR_{uuid.uuid4().hex[:8].upper()}"
        group = random.choice(peer_groups)

        # Baselines for USD
        stats = {
            "STUDENT": (40, 25),
            "RETAIL": (150, 90),
            "SME": (2500, 1200),
            "CORPORATE": (45000, 20000),
            "HNI": (8000, 3500)
        }
        avg, std = stats[group]

        customer_db[u_id] = {
            "user_id": u_id,
            "full_name": fake.name(),
            "national_id": fake.ssn(),
            "contact_id": fake.email(),
            "phone_number": fake.phone_number(),
            "occupation": random.choice(occupations),
            "physical_address": fake.address().replace('\n', ', '),
            "account_age_days": random.randint(1, 3650),
            "user_risk_score": random.randint(1, 100),
            "peer_group": group,
            "expected_monthly_vol": avg * 10,
            "historical_avg_txn": avg,
            "historical_std_dev": std,
            "known_counterparties_count": random.randint(0, 50),
            "primary_device": f"DVC-{uuid.uuid4().hex[:6].upper()}",
            "primary_ip": fake.ipv4()
        }

    # --- 2. GENERATE TRANSACTIONS ---
    logger.info(f"Phase 2/2: Writing transactions to {OUTPUT_FILE}...")

    fieldnames = [
        "transaction_id", "timestamp", "sender_id", "receiver_id", "amount", "currency",
        "device_id", "ip_address", "lat_long", "receiver_bank_code", "receiver_country", "channel",
        "sender_name", "sender_contact_id", "sender_phone",
        "receiver_name", "receiver_contact_id",
        "transaction_type", "mcc_code", "transaction_status", "receiver_risk_score",
        "sender_national_id", "sender_occupation", "sender_address",
        "user_id", "account_age_days", "user_risk_score", "peer_group",
        "expected_monthly_vol", "historical_avg_txn", "historical_std_dev", "known_counterparties"
    ]

    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        customer_ids = list(customer_db.keys())

        for i in tqdm(range(NUM_TRANSACTIONS)):
            s_id = random.choice(customer_ids)
            profile = customer_db[s_id]

            # Artificial Mule Hub (Entity 0 receives 10% of all traffic)
            r_id = customer_ids[0] if i % 10 == 0 else random.choice(customer_ids)
            r_profile = customer_db[r_id]

            # Normal amount based on user profile
            amount = max(5.0, round(np.random.normal(profile["historical_avg_txn"], profile["historical_std_dev"]), 2))

            # Injected Structuring (Just below $10,000)
            if i % 1000 == 0:
                amount = round(random.uniform(9850.00, 9995.00), 2)

            writer.writerow({
                "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                "timestamp": (START_DATE + timedelta(seconds=i * 30)).isoformat(),
                "sender_id": s_id,
                "receiver_id": r_id,
                "amount": amount,
                "currency": CURRENCY,
                "device_id": profile["primary_device"],
                "ip_address": profile["primary_ip"],
                "lat_long": f"{fake.latitude()},{fake.longitude()}",
                "receiver_bank_code": f"BK-{random.choice(['CHAS', 'WFGO', 'BOFA'])}",
                "receiver_country": random.choice(countries),
                "channel": random.choice(["MOBILE", "WIRE", "ATM"]),
                "sender_name": profile["full_name"],
                "sender_contact_id": profile["contact_id"],
                "sender_phone": profile["phone_number"],
                "receiver_name": r_profile["full_name"],
                "receiver_contact_id": r_profile["contact_id"],
                "transaction_type": "WIRE_TRANSFER" if amount > 5000 else "P2P",
                "mcc_code": random.choice([5411, 5812, 6011]),
                "transaction_status": "COMPLETED",
                "receiver_risk_score": r_profile["user_risk_score"],
                "sender_national_id": profile["national_id"],
                "sender_occupation": profile["occupation"],
                "sender_address": profile["physical_address"],
                "user_id": profile["user_id"],
                "account_age_days": profile["account_age_days"],
                "user_risk_score": profile["user_risk_score"],
                "peer_group": profile["peer_group"],
                "expected_monthly_vol": profile["expected_monthly_vol"],
                "historical_avg_txn": profile["historical_avg_txn"],
                "historical_std_dev": profile["historical_std_dev"],
                "known_counterparties": profile["known_counterparties_count"]
            })

    logger.info("Generation Complete!")


if __name__ == "__main__":
    generate_tms_data()