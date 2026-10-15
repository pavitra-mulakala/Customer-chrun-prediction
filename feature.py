# =========================================
# FEATURE REPRESENTATION (DATA DICTIONARY)
# =========================================

import pandas as pd

# Load dataset
# NOTE: this previously pointed at "Telco-Customer-Churn.csv" (the original
# Kaggle file, with PascalCase columns like "customerID"/"MonthlyCharges").
# Every other script in this project reads "telco_churn_dataset.csv", which
# uses snake_case columns instead — so this file was out of sync with the
# rest of the pipeline and its descriptions didn't match any real column.
df = pd.read_csv("telco_churn_dataset.csv")

# Feature descriptions (MANUALLY DEFINED) - matches telco_churn_dataset.csv
feature_description = {
    "customer_id": "Unique identifier assigned to each customer.",
    "service_provider": "Telecom provider the customer is with (e.g. Jio, Airtel).",
    "gender": "Customer's gender (Male/Female).",
    "age": "Customer's age in years.",
    "tenure": "Number of months the customer has stayed with the company.",
    "internet_type": "Type of internet service (DSL, Fiber, etc.).",
    "contract": "Type of contract (Month-to-month, One year, Two year).",
    "payment_method": "Customer's payment method (Debit card, Net banking, UPI, etc.).",
    "monthly_charge": "Amount charged to the customer per month.",
    "total_charges": "Total amount charged to the customer since joining.",
    "satisfaction_score": "Customer's self-reported satisfaction rating.",
    "tech_support": "Whether the customer has technical support service (Yes/No).",
    "online_security": "Whether the customer has an online security add-on (Yes/No).",
    "streaming_tv": "Whether the customer has TV streaming service (Yes/No).",
    "paperless_billing": "Whether the customer uses paperless billing (Yes/No).",
    "churn": "Indicates whether the customer left the company (Yes/No).",
    "churn_category": "High-level category for why the customer churned.",
    "churn_reason": "Specific stated reason the customer churned."
}

print("\n================ FEATURE REPRESENTATION ================\n")

for col in df.columns:
    print(f"Feature Name: {col}")
    print(f"Description : {feature_description.get(col, 'No description available.')}")
    print("-" * 60)