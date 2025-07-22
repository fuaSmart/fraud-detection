import pandas as pd
import os

def load_fraud_data(file_path):
    """Loads the fraud transaction data."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {os.path.basename(file_path)}.")
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        print(f"An error occurred while loading {os.path.basename(file_path)}: {e}")
        return None

def load_ip_to_country_data(file_path):
    """Loads the IP address to country mapping data."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {os.path.basename(file_path)}.")

        df['lower_bound_ip_address'] = df['lower_bound_ip_address'].astype(int)
        df['upper_bound_ip_address'] = df['upper_bound_ip_address'].astype(int)

        df = df.sort_values(by='lower_bound_ip_address').reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        print(f"An error occurred while loading {os.path.basename(file_path)}: {e}")
        return None

def load_creditcard_data(file_path):
    """Loads the credit card transaction data."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {os.path.basename(file_path)}.")
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        print(f"An error occurred while loading {os.path.basename(file_path)}: {e}")
        return None

if __name__ == '__main__':

    RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

    fraud_data_path = os.path.join(RAW_DATA_PATH, 'Fraud_Data.csv')
    ip_to_country_path = os.path.join(RAW_DATA_PATH, 'IpAddress_to_Country.csv')
    creditcard_data_path = os.path.join(RAW_DATA_PATH, 'creditcard.csv')

    fraud_df = load_fraud_data(fraud_data_path)
    ip_df = load_ip_to_country_data(ip_to_country_path)
    creditcard_df = load_creditcard_data(creditcard_data_path)

    if fraud_df is not None:
        print("\nFraud data head (from data_ingestion.py test):")
        print(fraud_df.head())
    if ip_df is not None:
        print("\nIP to Country data head (from data_ingestion.py test):")
        print(ip_df.head())
    if creditcard_df is not None:
        print("\nCredit Card data head (from data_ingestion.py test):")
        print(creditcard_df.head())