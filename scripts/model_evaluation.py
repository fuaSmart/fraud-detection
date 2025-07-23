import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve 
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve


def load_model(file_path):
    """Loads a trained model."""
    try:
        model = joblib.load(file_path)
        print(f"Successfully loaded model from {os.path.basename(file_path)}.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file {file_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model from {os.path.basename(file_path)}: {e}")
        return None

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained model and prints key performance metrics.
    Generates ROC and Precision-Recall curves.
    """
    if model is None or X_test is None or y_test is None:
        print(f"Skipping evaluation for {model_name}: Missing model or test data.")
        return

    print(f"\n--- Evaluating {model_name} ---")

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] 

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Non-Fraud', 'Predicted Fraud'],
                yticklabels=['Actual Non-Fraud', 'Actual Fraud'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()


    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba) 
    plt.plot(fpr, tpr, label=f'{model_name} (ROC AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba) 
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (PR AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()

    print(f"--- {model_name} Evaluation Complete ---")


if __name__ == '__main__':
    
    BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
    MODELS_PATH = os.path.join(BASE_DIR, 'models')


    try:
        dummy_fraud_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'fraud_data_processed.csv'))
        
        features_to_exclude_fraud = [
            'user_id', 'device_id', 'signup_time', 'purchase_time',
            'ip_address', 'ip_address_int', 'country', 'class'
        ]
        X_cols_fraud_initial = [col for col in dummy_fraud_df.columns if col not in features_to_exclude_fraud and not col.startswith(('source_', 'browser_', 'sex_'))]
        X_cols_fraud_initial.extend([col for col in dummy_fraud_df.columns if col.startswith(('source_', 'browser_', 'sex_'))])
        X_cols_fraud_final = [col for col in X_cols_fraud_initial if pd.api.types.is_numeric_dtype(dummy_fraud_df[col])]

        X_fraud = dummy_fraud_df[X_cols_fraud_final]
        y_fraud = dummy_fraud_df['class']

        _, X_test_fraud, _, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)

        # Load the trained fraud model
        fraud_model_path = os.path.join(MODELS_PATH, 'fraud_detection_rf_model.joblib')
        trained_fraud_pipeline = load_model(fraud_model_path)
        
        # Evaluate the fraud model
        if trained_fraud_pipeline:
            evaluate_model(trained_fraud_pipeline, X_test_fraud, y_test_fraud, model_name="Fraud Detection RandomForest")

    except FileNotFoundError:
        print("\nCould not find processed data or trained model files. Please run `python main.py` first.")
    except Exception as e:
        print(f"\nAn error occurred during independent evaluation: {e}")

    try:
        # Re-create X_test and y_test structure for credit card data
        dummy_creditcard_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'creditcard_data_processed.csv'))
        X_creditcard = dummy_creditcard_df.drop('Class', axis=1)
        y_creditcard = dummy_creditcard_df['Class']

        _, X_test_creditcard, _, y_test_creditcard = train_test_split(X_creditcard, y_creditcard, test_size=0.2, random_state=42, stratify=y_creditcard)

        # Load the trained credit card model
        creditcard_model_path = os.path.join(MODELS_PATH, 'creditcard_lr_model.joblib')
        trained_creditcard_pipeline = load_model(creditcard_model_path)

        # Evaluate the credit card model
        if trained_creditcard_pipeline:
            evaluate_model(trained_creditcard_pipeline, X_test_creditcard, y_test_creditcard, model_name="Credit Card Logistic Regression")

    except FileNotFoundError:
        print("\nCould not find processed data or trained model files.")
    except Exception as e:
        print(f"\nAn error occurred during independent evaluation: {e}")