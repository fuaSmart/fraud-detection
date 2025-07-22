import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from data_ingestion import load_fraud_data, load_ip_to_country_data, load_creditcard_data
from data_preprocessing import clean_fraud_data, clean_creditcard_data, merge_ip_data
from feature_engineering import create_fraud_features, create_creditcard_features

def main():
 
    # Define paths
    BASE_DIR = os.path.dirname(__file__)
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    print(f"Ensured processed data directory exists at: {PROCESSED_DATA_PATH}")

    # ---  Data Ingestion ---
    print("\n--- Starting Data Ingestion ---")
    fraud_data_raw = load_fraud_data(os.path.join(RAW_DATA_PATH, 'Fraud_Data.csv'))
    ip_to_country_raw = load_ip_to_country_data(os.path.join(RAW_DATA_PATH, 'IpAddress_to_Country.csv'))
    creditcard_data_raw = load_creditcard_data(os.path.join(RAW_DATA_PATH, 'creditcard.csv'))
    print("--- Data Ingestion Complete ---")

    # --- Data Preprocessing ---
    print("\n--- Starting Data Preprocessing ---")
    if fraud_data_raw is not None:
        fraud_data_cleaned = clean_fraud_data(fraud_data_raw.copy())
        # Merge IP data
        fraud_data_preprocessed = merge_ip_data(fraud_data_cleaned, ip_to_country_raw)
    else:
        fraud_data_preprocessed = None
        print("Skipping fraud data preprocessing due to ingestion failure.")

    if creditcard_data_raw is not None:
        creditcard_data_preprocessed = clean_creditcard_data(creditcard_data_raw.copy())
    else:
        creditcard_data_preprocessed = None
        print("Skipping credit card data preprocessing due to ingestion failure.")
    print("--- Data Preprocessing Complete ---")

    # --- Feature Engineering ---
    print("\n--- Starting Feature Engineering ---")
    if fraud_data_preprocessed is not None:
        fraud_data_featured = create_fraud_features(fraud_data_preprocessed.copy())
    else:
        fraud_data_featured = None
        print("Skipping fraud data feature engineering due to preprocessing failure.")

    if creditcard_data_preprocessed is not None:
        creditcard_data_featured = create_creditcard_features(creditcard_data_preprocessed.copy())
    else:
        creditcard_data_featured = None
        print("Skipping credit card data feature engineering due to preprocessing failure.")
    print("--- Feature Engineering Complete ---")

    # --- Save Processed Data ---
    print("\n--- Saving Processed Data ---")
    if fraud_data_featured is not None:
        output_path_fraud = os.path.join(PROCESSED_DATA_PATH, 'fraud_data_processed.csv')
        fraud_data_featured.to_csv(output_path_fraud, index=False)
        print(f"Processed fraud data saved to: {output_path_fraud}")
    else:
        print("No fraud data to save.")

    if creditcard_data_featured is not None:
        output_path_creditcard = os.path.join(PROCESSED_DATA_PATH, 'creditcard_data_processed.csv')
        creditcard_data_featured.to_csv(output_path_creditcard, index=False)
        print(f"Processed credit card data saved to: {output_path_creditcard}")
    else:
        print("No credit card data to save.")
    print("--- Data Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()