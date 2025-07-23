import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from data_ingestion import load_fraud_data, load_ip_to_country_data, load_creditcard_data
from data_preprocessing import clean_fraud_data, clean_creditcard_data, merge_ip_data 
from feature_engineering import create_fraud_features, create_creditcard_features
from model_training import train_fraud_model, train_creditcard_model
from model_evaluation import evaluate_model

def main():
    """
    Main function to run the fraud detection pipeline.
    """
    print("--- Starting Fraud Detection Pipeline ---")

    base_dir = os.path.dirname(__file__)
    data_raw_path = os.path.join(base_dir, 'data', 'raw')
    data_processed_path = os.path.join(base_dir, 'data', 'processed')
    models_path = os.path.join(base_dir, 'models')

    os.makedirs(data_processed_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # --- 1. Data Ingestion ---
    print("\n--- Data Ingestion ---")
    fraud_data_raw_path = os.path.join(data_raw_path, 'Fraud_Data.csv')
    ip_data_raw_path = os.path.join(data_raw_path, 'IpAddress_to_Country.csv')
    creditcard_data_raw_path = os.path.join(data_raw_path, 'creditcard.csv')

    fraud_df = load_fraud_data(fraud_data_raw_path)
    ip_df = load_ip_to_country_data(ip_data_raw_path)
    creditcard_df = load_creditcard_data(creditcard_data_raw_path)

    # --- 2. Data Preprocessing ---
    print("\n--- Data Preprocessing ---")
    if fraud_df is not None:
        fraud_df_cleaned = clean_fraud_data(fraud_df.copy()) 
        if ip_df is not None:
            fraud_df_preprocessed = merge_ip_data(fraud_df_cleaned, ip_df)
        else:
            fraud_df_preprocessed = fraud_df_cleaned 
            print("Skipping IP data merge for fraud data due to missing IP dataframe.")
    else:
        fraud_df_preprocessed = None
        print("Skipping fraud data preprocessing due to missing dataframe.")

    if creditcard_df is not None:
        creditcard_df_preprocessed = clean_creditcard_data(creditcard_df.copy()) 
    else:
        creditcard_df_preprocessed = None
        print("Skipping credit card data preprocessing due to missing dataframe.")

    print("--- Data Preprocessing Complete ---")


    # --- 3. Feature Engineering ---
    print("\n--- Feature Engineering ---")
    fraud_df_featured = None
    creditcard_df_featured = None

    if fraud_df_preprocessed is not None:
        fraud_df_featured = create_fraud_features(fraud_df_preprocessed.copy())
        print("Feature engineering complete for fraud data.")
    else:
        print("Skipping fraud data feature engineering due to missing preprocessed dataframe.")

    if creditcard_df_preprocessed is not None:
        creditcard_df_featured = create_creditcard_features(creditcard_df_preprocessed.copy())
        print("Feature engineering complete for credit card data.")
    else:
        print("Skipping credit card data feature engineering due to missing preprocessed dataframe.")
    print("--- Feature Engineering Complete ---")


    # --- 4. Saving Processed Data ---
    print("\n--- Saving Processed Data ---")
    if fraud_df_featured is not None:
        processed_fraud_path = os.path.join(data_processed_path, 'fraud_data_processed.csv')
        fraud_df_featured.to_csv(processed_fraud_path, index=False)
        print(f"Processed fraud data saved to: {processed_fraud_path}")
    if creditcard_df_featured is not None:
        processed_creditcard_path = os.path.join(data_processed_path, 'creditcard_data_processed.csv')
        creditcard_df_featured.to_csv(processed_creditcard_path, index=False)
        print(f"Processed credit card data saved to: {processed_creditcard_path}")
    print("--- Data Pipeline Execution Complete ---")


    # --- 5. Model Training ---
    print("\n--- Model Training ---")
    trained_fraud_pipeline = None
    X_test_fraud = None
    y_test_fraud = None
    trained_creditcard_pipeline = None
    X_test_creditcard = None
    y_test_creditcard = None

    if fraud_df_featured is not None:
        fraud_model_save_path = os.path.join(models_path, 'fraud_detection_rf_model.joblib')
        trained_fraud_pipeline, X_test_fraud, y_test_fraud = train_fraud_model(
            fraud_df_featured.copy(),
            model_name="RandomForest",
            save_path=fraud_model_save_path
        )
    else:
        print("Skipping fraud model training due to missing fraud data.")

    if creditcard_df_featured is not None:
        creditcard_model_save_path = os.path.join(models_path, 'creditcard_lr_model.joblib')
        trained_creditcard_pipeline, X_test_creditcard, y_test_creditcard = train_creditcard_model(
            creditcard_df_featured.copy(),
            model_name="LogisticRegression",
            save_path=creditcard_model_save_path
        )
    else:
        print("Skipping credit card model training due to missing credit card data.")
    print("--- Model Training Complete ---")


    # --- 6. Model Evaluation ---
    print("\n--- Model Evaluation ---")
    if trained_fraud_pipeline is not None and X_test_fraud is not None and y_test_fraud is not None:
        evaluate_model(trained_fraud_pipeline, X_test_fraud, y_test_fraud, model_name="Fraud Detection RandomForest")
    else:
        print("Skipping fraud model evaluation due to missing model or test data.")

    if trained_creditcard_pipeline is not None and X_test_creditcard is not None and y_test_creditcard is not None:
        evaluate_model(trained_creditcard_pipeline, X_test_creditcard, y_test_creditcard, model_name="Credit Card Logistic Regression")
    else:
        print("Skipping credit card model evaluation due to missing model or test data.")
    print("--- Model Evaluation Complete ---")

    print("\n--- Pipeline Execution Finished ---")

if __name__ == "__main__":
    main()