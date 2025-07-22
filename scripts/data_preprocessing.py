import pandas as pd
import numpy as np
import os

def clean_fraud_data(df):

    if df is None:
        return None

    print("\n--- Cleaning Fraud Data ---")
    # Impute missing 'sex' values with the mode
    if 'sex' in df.columns and df['sex'].isnull().any():
        mode_sex = df['sex'].mode()[0]
        df['sex'].fillna(mode_sex, inplace=True)
        print(f"Missing 'sex' values imputed with mode: {mode_sex}")

    if 'age' in df.columns and df['age'].isnull().any():
        median_age = df['age'].median()
        df['age'].fillna(median_age, inplace=True)
        print(f"Missing 'age' values imputed with median: {median_age}")

    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        print("'signup_time' converted to datetime.")
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        print("'purchase_time' converted to datetime.")

    # Remove duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(subset=['user_id', 'device_id', 'purchase_time', 'purchase_value'], inplace=True, keep='first')
    rows_after_dedup = df.shape[0]
    if initial_rows - rows_after_dedup > 0:
        print(f"Removed {initial_rows - rows_after_dedup} duplicate rows from fraud data.")
    else:
        print("No duplicate rows found in fraud data.")

    print("Fraud data cleaning complete.")
    return df

def clean_creditcard_data(df):
    """Performs initial cleaning for creditcard.csv."""
    if df is None:
        return None

    print("\n--- Cleaning Credit Card Data ---")

    initial_rows = df.shape[0]
    df.drop_duplicates(subset=[col for col in df.columns if col != 'Class'], inplace=True, keep='first')
    rows_after_dedup = df.shape[0]
    if initial_rows - rows_after_dedup > 0:
        print(f"Removed {initial_rows - rows_after_dedup} duplicate rows from credit card data.")
    else:
        print("No duplicate rows found in credit card data.")

    print("Credit card data cleaning complete.")
    return df

def ip_to_int(ip_address):
    """Converts a numerical IP address (float) to an integer."""
    if pd.isna(ip_address):
        return np.nan
    try:
        return int(ip_address)
    except (ValueError, TypeError):
        return np.nan

def get_country(ip_int, ip_country_df):
    """
    Looks up the country for a given IP integer using the IP to country DataFrame.
    Assumes ip_country_df is sorted by 'lower_bound_ip_address'.
    """
    if pd.isna(ip_int) or ip_country_df is None:
        return 'Unknown'
    
    idx = ip_country_df['lower_bound_ip_address'].searchsorted(ip_int, side='right') - 1
    
    if idx >= 0 and idx < len(ip_country_df) and \
       ip_int >= ip_country_df.loc[idx, 'lower_bound_ip_address'] and \
       ip_int <= ip_country_df.loc[idx, 'upper_bound_ip_address']:
        return ip_country_df.loc[idx, 'country']
    return 'Unknown'

def merge_ip_data(fraud_df, ip_country_df):
    """
    Merges fraud_df with country information from ip_country_df based on IP address.
    Adds 'ip_address_int' and 'country' columns to fraud_df.
    """
    if fraud_df is None:
        print("Fraud DataFrame is None, skipping IP data merge.")
        return None
    if ip_country_df is None:
        print("IP to Country DataFrame is None, skipping IP data merge.")
        fraud_df['ip_address_int'] = np.nan 
        fraud_df['country'] = 'Unknown' 
        return fraud_df

    print("\nConverting IP addresses to integer format in fraud data...")
    fraud_df['ip_address_int'] = fraud_df['ip_address'].apply(ip_to_int)
    print("IP address conversion complete.")

    print("Merging Fraud Data with IpAddress_to_Country...")

    fraud_df['country'] = fraud_df['ip_address_int'].apply(lambda x: get_country(x, ip_country_df))
    print("Merge complete.")
    return fraud_df

if __name__ == '__main__':

    # Create dummy data for testing purposes
    dummy_fraud_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'signup_time': ['2015-02-24 22:55:49', '2015-06-07 20:39:50', '2015-01-01 18:52:44', '2015-04-28 21:13:25', '2015-07-21 07:09:52'],
        'purchase_time': ['2015-04-18 02:47:11', '2015-06-08 01:38:54', '2015-01-01 18:52:45', '2015-05-04 13:54:50', '2015-09-09 18:40:53'],
        'purchase_value': [34, 16, 15, 44, 39],
        'device_id': ['QVPSPJUOCKZAR', 'EOGFQPIZPYXFZ', 'YSSKYOSJHPPLJ', 'ATGTXKYKUDUQN', 'NAUITBZFJKHWW'],
        'source': ['SEO', 'Ads', 'SEO', 'SEO', 'Ads'],
        'browser': ['Chrome', 'Chrome', 'Opera', 'Safari', 'Safari'],
        'sex': ['M', 'F', 'M', 'M', 'M'],
        'age': [39, 53, 53, 41, 45],
        'ip_address': [732758368.79972, 350311387.865908, 2621473820.11095, 3840542443.91396, 415583117.452712],
        'class': [0, 0, 1, 0, 0]
    })
    dummy_ip_to_country = pd.DataFrame({
        'lower_bound_ip_address': [16777216, 350311387, 732758368, 2621473820, 3840542443],
        'upper_bound_ip_address': [16777471, 350311388, 732758369, 2621473821, 3840542444],
        'country': ['Australia', 'United States', 'Canada', 'Germany', 'France']
    })
    dummy_creditcard_data = pd.DataFrame({
        'Time': [0, 1, 2, 3, 4],
        'V1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'V2': [0.4, 0.5, 0.6, 0.7, 0.8],
        'Amount': [10.0, 20.0, 30.0, 15.0, 25.0],
        'Class': [0, 0, 1, 0, 1]
    })

    # Testing functions
    processed_fraud_df = clean_fraud_data(dummy_fraud_data.copy())
    processed_creditcard_df = clean_creditcard_data(dummy_creditcard_data.copy())
    
    processed_ip_to_country_df = dummy_ip_to_country.sort_values(by='lower_bound_ip_address').reset_index(drop=True)

    final_fraud_df = merge_ip_data(processed_fraud_df, processed_ip_to_country_df)

    if final_fraud_df is not None:
        print("\nProcessed Fraud Data Head (from data_preprocessing.py test):")
        print(final_fraud_df[['ip_address', 'ip_address_int', 'country', 'class']].head())

    if processed_creditcard_df is not None:
        print("\nProcessed Credit Card Data Head (from data_preprocessing.py test):")
        print(processed_creditcard_df.head())