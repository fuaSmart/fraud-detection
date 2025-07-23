import pandas as pd
import os
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline as ImbPipeline 
from sklearn.pipeline import Pipeline 

def load_processed_data(file_path):
    """Loads a processed dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded processed data from {os.path.basename(file_path)}.")
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        print(f"An error occurred while loading {os.path.basename(file_path)}: {e}")
        return None

def train_fraud_model(df, model_name="RandomForest", save_path=None):
    """
    Trains a fraud detection model using the processed fraud data.
    Applies SMOTE for class imbalance and saves the trained model.
    """
    if df is None:
        print("Fraud DataFrame is None, skipping model training.")
        return None, None

    print(f"\n--- Training Fraud Model ({model_name}) ---")

  
    features_to_exclude = [
        'user_id', 'device_id', 'signup_time', 'purchase_time',
        'ip_address', 'ip_address_int', 'country', 'class'
    ]
    
    # Dynamically determine columns to use
    X_cols = [col for col in df.columns if col not in features_to_exclude and not col.startswith(('source_', 'browser_', 'sex_'))]
    
    X_cols.extend([col for col in df.columns if col.startswith(('source_', 'browser_', 'sex_'))])
    
    X_cols_final = [col for col in X_cols if pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[X_cols_final]
    y = df['class']

    print(f"Features used for fraud model: {X.columns.tolist()}")
    print(f"Target variable distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Define the model
    if model_name == "LogisticRegression":
        model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') 
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42) 
    else:
        print(f"Model '{model_name}' not recognized. Using RandomForestClassifier.")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Create a pipeline with scaling and SMOTE
    pipeline_steps = [
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ]
    

    if model_name == "GradientBoosting":
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ]
    
    if model_name in ["LogisticRegression", "RandomForest"]:
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)), 
            ('classifier', model)
        ]

    pipeline = ImbPipeline(steps=pipeline_steps)

    # Train the pipeline
    print(f"Training {model_name} pipeline with SMOTE...")
    pipeline.fit(X_train, y_train)
    print(f"{model_name} training complete.")

    # Save the trained model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pipeline, save_path)
        print(f"Trained fraud model saved to: {save_path}")

    return pipeline, X_test, y_test

def train_creditcard_model(df, model_name="LogisticRegression", save_path=None):
    """
    Trains a credit card fraud detection model.
    Assumes 'Time', 'Amount', V-features are numeric.
    """
    if df is None:
        print("Credit Card DataFrame is None, skipping model training.")
        return None, None

    print(f"\n--- Training Credit Card Model ({model_name}) ---")

    # Define features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    print(f"Features used for credit card model: {X.columns.tolist()}")
    print(f"Target variable distribution:\n{y.value_counts()}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Define the model Logistic Regression 
    if model_name == "LogisticRegression":
        model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    else:
        print(f"Model '{model_name}' not recognized. Using LogisticRegression.")
        model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)

    # Create a pipeline with scaling and SMOTE
    pipeline_steps = [
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)), 
        ('classifier', model)
    ]

    pipeline = ImbPipeline(steps=pipeline_steps)

    # Train the pipeline
    print(f"Training {model_name} pipeline with SMOTE...")
    pipeline.fit(X_train, y_train)
    print(f"{model_name} training complete.")

    # Save the trained model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pipeline, save_path)
        print(f"Trained credit card model saved to: {save_path}")

    return pipeline, X_test, y_test


if __name__ == '__main__':
    
    # Define paths based on expected project structure
    BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
    MODELS_PATH = os.path.join(BASE_DIR, 'models')

    # Load processed data
    fraud_df = load_processed_data(os.path.join(PROCESSED_DATA_PATH, 'fraud_data_processed.csv'))
    creditcard_df = load_processed_data(os.path.join(PROCESSED_DATA_PATH, 'creditcard_data_processed.csv'))

    # Train fraud detection model
    if fraud_df is not None:
        fraud_model_path = os.path.join(MODELS_PATH, 'fraud_detection_rf_model.joblib')
        trained_fraud_pipeline, X_test_fraud, y_test_fraud = train_fraud_model(
            fraud_df.copy(), model_name="RandomForest", save_path=fraud_model_path
        )

    # Train credit card fraud detection model
    if creditcard_df is not None:
        creditcard_model_path = os.path.join(MODELS_PATH, 'creditcard_lr_model.joblib')
        trained_creditcard_pipeline, X_test_creditcard, y_test_creditcard = train_creditcard_model(
            creditcard_df.copy(), model_name="LogisticRegression", save_path=creditcard_model_path
        )
