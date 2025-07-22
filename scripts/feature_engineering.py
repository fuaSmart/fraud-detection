import pandas as pd
import numpy as np
import os

def create_fraud_features(df):
  
    if df is None:
        return None

    print("\n--- Creating Fraud Features ---")

    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        df['time_difference'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        print("Feature 'time_difference' created.")
    else:
        print("Warning: 'signup_time' or 'purchase_time' not found. Skipping 'time_difference'.")

    if 'signup_time' in df.columns:
        df['signup_day_of_week'] = df['signup_time'].dt.dayofweek
        print("Feature 'signup_day_of_week' created.")
    if 'purchase_time' in df.columns:
        df['purchase_day_of_week'] = df['purchase_time'].dt.dayofweek
        print("Feature 'purchase_day_of_week' created.")

    if 'signup_time' in df.columns:
        df['signup_hour'] = df['signup_time'].dt.hour
        print("Feature 'signup_hour' created.")
    if 'purchase_time' in df.columns:
        df['purchase_hour'] = df['purchase_time'].dt.hour
        print("Feature 'purchase_hour' created.")

    categorical_cols = ['source', 'browser', 'sex']
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
            print(f"One-hot encoded '{col}'.")
        else:
            print(f"Warning: Categorical column '{col}' not found. Skipping encoding.")

   
    df['purchase_value_per_age'] = df['purchase_value'] / (df['age'] + 1e-6) 
    print("Feature 'purchase_value_per_age' created.")

    
    device_id_counts = df.groupby('device_id')['user_id'].transform('count')
    df['device_id_user_count'] = device_id_counts
    print("Feature 'device_id_user_count' created.")


    print("Feature engineering complete for fraud data.")
    return df


def create_creditcard_features(df):
    """
    Generates features for the credit card transaction DataFrame.
    """
    if df is None:
        return None

    print("\n--- Creating Credit Card Features ---")
  
    print("Feature engineering complete for credit card data (placeholder).")
    return df

if __name__ == '__main__':
   
    dummy_processed_fraud_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'signup_time': pd.to_datetime(['2015-02-24 22:55:49', '2015-06-07 20:39:50', '2015-01-01 18:52:44', '2015-04-28 21:13:25', '2015-07-21 07:09:52']),
        'purchase_time': pd.to_datetime(['2015-04-18 02:47:11', '2015-06-08 01:38:54', '2015-01-01 18:52:45', '2015-05-04 13:54:50', '2015-09-09 18:40:53']),
        'purchase_value': [34, 16, 15, 44, 39],
        'device_id': ['deviceA', 'deviceB', 'deviceA', 'deviceC', 'deviceB'], # Duplicates for testing counts
        'source': ['SEO', 'Ads', 'SEO', 'Direct', 'Ads'],
        'browser': ['Chrome', 'Chrome', 'Opera', 'Safari', 'Chrome'],
        'sex': ['M', 'F', 'M', 'M', 'F'],
        'age': [39, 53, 53, 41, 45],
        'ip_address': [732758368.0, 350311387.0, 2621473820.0, 3840542443.0, 415583117.0],
        'ip_address_int': [732758368, 350311387, 2621473820, 3840542443, 415583117],
        'country': ['Canada', 'United States', 'Germany', 'France', 'Unknown'],
        'class': [0, 0, 1, 0, 0]
    })
    dummy_processed_creditcard_data = pd.DataFrame({
        'Time': [0, 1, 2, 3, 4],
        'V1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'V2': [0.4, 0.5, 0.6, 0.7, 0.8],
        'Amount': [10.0, 20.0, 30.0, 15.0, 25.0],
        'Class': [0, 0, 1, 0, 1]
    })

    # Test the functions
    featured_fraud_df = create_fraud_features(dummy_processed_fraud_data.copy())
    featured_creditcard_df = create_creditcard_features(dummy_processed_creditcard_data.copy())

    if featured_fraud_df is not None:
        print("\nFeatured Fraud Data Head (from feature_engineering.py test):")
        print(featured_fraud_df.head())
        print("\nFeatured Fraud Data Info (from feature_engineering.py test):")
        featured_fraud_df.info()

    if featured_creditcard_df is not None:
        print("\nFeatured Credit Card Data Head (from feature_engineering.py test):")
        print(featured_creditcard_df.head())