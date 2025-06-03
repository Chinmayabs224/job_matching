# src/acceptance_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import os

# --- Configuration ---
ACCEPTANCE_MODEL_SAVE_PATH = "models/acceptance_predictor_logreg.joblib"

# --- Model Training Function ---
def train_acceptance_predictor(candidate_job_features_df, target_column='accepted_offer'):
    """
    Trains a Logistic Regression model to predict candidate acceptance probability.

    Args:
        candidate_job_features_df (pd.DataFrame): DataFrame with features and the target variable.
            Example features: 'match_score', 'salary_diff_percentage', 'candidate_experience_years', 'is_remote'.
            Target variable ('accepted_offer') should be binary (1 for accepted, 0 for rejected).
        target_column (str): Name of the binary target column.

    Returns:
        sklearn.pipeline.Pipeline: The trained classification pipeline.
        dict: Training and test metrics.
    """
    if not isinstance(candidate_job_features_df, pd.DataFrame) or target_column not in candidate_job_features_df.columns:
        print("Error: Invalid input DataFrame or target column name.")
        return None, None

    # Define feature types (example)
    numerical_features = ['match_score', 'salary_diff_percentage', 'candidate_experience_years']
    categorical_features = [] # e.g., 'job_category_classified', 'location_preference_match'
    binary_features = ['is_remote'] # Features already 0/1

    # Ensure all necessary columns exist
    # For simplicity, assume these columns are present. Real-world would need robust checking.
    current_numerical_features = [f for f in numerical_features if f in candidate_job_features_df.columns]
    current_categorical_features = [f for f in categorical_features if f in candidate_job_features_df.columns]
    current_binary_features = [f for f in binary_features if f in candidate_job_features_df.columns]
    
    feature_columns = current_numerical_features + current_categorical_features + current_binary_features
    if not feature_columns:
        print("Error: No feature columns identified or present in the DataFrame.")
        return None, None

    required_cols = feature_columns + [target_column]
    missing_cols = [col for col in required_cols if col not in candidate_job_features_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in DataFrame: {missing_cols}")
        return None, None

    # Handle NaNs (simple drop for this example)
    candidate_job_features_df = candidate_job_features_df.dropna(subset=required_cols)
    if candidate_job_features_df.empty or len(candidate_job_features_df) < 10:
        print("Error: DataFrame is empty or has insufficient data after dropping NaNs.")
        return None, None
    
    if candidate_job_features_df[target_column].nunique() < 2:
        print(f"Error: Target column '{target_column}' must have at least two unique classes for logistic regression.")
        return None, None

    X = candidate_job_features_df[feature_columns]
    y = candidate_job_features_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Preprocessing
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    transformers_list = []
    if current_numerical_features:
        transformers_list.append(('num', numerical_transformer, current_numerical_features))
    if current_categorical_features:
        transformers_list.append(('cat', categorical_transformer, current_categorical_features))
    
    # For binary features that are already 0/1, we can use 'passthrough' or ensure they are numerical
    # If they are not part of 'num' or 'cat', they need to be handled by 'remainder'
    # For simplicity, if binary_features are present, ensure they are included and correctly typed.
    # If binary_features are the only features, ColumnTransformer might not be needed or needs specific handling.

    if not transformers_list: # Only binary features or no features to transform
        if current_binary_features:
             # If only binary, ensure they are numeric and pass them through or scale if necessary
            preprocessor = ColumnTransformer(transformers=[('bin', StandardScaler(), current_binary_features)], remainder='drop') 
            # Or 'passthrough' if they are fine as is and don't need scaling.
        else: # No features to transform, this case should ideally not happen if feature_columns is not empty
            print("Warning: No features specified for transformation in preprocessor.")
            # Create a dummy preprocessor that does nothing if no specific transformers are needed
            preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')
    else:
        preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough') # 'passthrough' for binary_features if not in num/cat

    # Model
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    print("Training the acceptance predictor...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1] # Probability of acceptance

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_roc_auc = roc_auc_score(y_test, y_proba_test)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}, Test ROC AUC: {test_roc_auc:.4f}")
    print("\nTest Set Classification Report:")
    report_str = classification_report(y_test, y_pred_test, zero_division=0)
    print(report_str)
    print("\nTest Set Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))

    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_roc_auc": test_roc_auc,
        "classification_report_test": classification_report(y_test, y_pred_test, output_dict=True, zero_division=0),
        "confusion_matrix_test": confusion_matrix(y_test, y_pred_test).tolist()
    }

    os.makedirs(os.path.dirname(ACCEPTANCE_MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(pipeline, ACCEPTANCE_MODEL_SAVE_PATH)
    print(f"Trained acceptance prediction pipeline saved to {ACCEPTANCE_MODEL_SAVE_PATH}")

    return pipeline, metrics

# --- Prediction Function ---
def predict_acceptance_probability(features_df, model_pipeline=None):
    """
    Predicts the probability of a candidate accepting an offer.

    Args:
        features_df (pd.DataFrame): DataFrame with features for prediction.
        model_pipeline (sklearn.pipeline.Pipeline, optional): Trained model pipeline. Loads if None.

    Returns:
        np.ndarray: Array of acceptance probabilities (between 0 and 1).
                    Returns None if an error occurs.
    """
    if model_pipeline is None:
        try:
            model_pipeline = joblib.load(ACCEPTANCE_MODEL_SAVE_PATH)
            print(f"Loaded acceptance prediction pipeline from {ACCEPTANCE_MODEL_SAVE_PATH}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {ACCEPTANCE_MODEL_SAVE_PATH}. Train first.")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    if not isinstance(features_df, pd.DataFrame) or features_df.empty:
        print("Error: Input for prediction must be a non-empty DataFrame.")
        return None
    
    # Ensure columns match training (handled by pipeline's preprocessor if columns are missing/extra and 'remainder' is set)
    try:
        probabilities = model_pipeline.predict_proba(features_df)[:, 1] # Probability of class 1 (accepted)
        return probabilities
    except Exception as e:
        print(f"Error during acceptance probability prediction: {e}")
        # This can happen if features_df doesn't have columns the preprocessor expects.
        # E.g. if a column specified in `numerical_features` during training is missing here.
        return None

# --- Sample Usage ---
if __name__ == "__main__":
    print("Starting Candidate Acceptance Prediction Module...")

    # Sample data (highly simplified)
    # 'salary_diff_percentage': (offered_salary - predicted_market_salary) / predicted_market_salary * 100
    # 'accepted_offer': 1 if accepted, 0 if rejected
    data = {
        'candidate_id': [f'cand{i}' for i in range(1, 51)],
        'match_score': np.random.rand(50) * 0.5 + 0.3,  # Range 0.3 to 0.8
        'salary_diff_percentage': np.random.randn(50) * 10, # Mean 0, std 10
        'candidate_experience_years': np.random.randint(1, 15, 50),
        'is_remote': np.random.randint(0, 2, 50), # 0 for no, 1 for yes
        'accepted_offer': np.random.randint(0, 2, 50) # Random target for example
    }
    sample_acceptance_df = pd.DataFrame(data)
    
    # Make the target somewhat correlated for a more meaningful (though still dummy) example
    # Higher match score, positive salary diff, and remote option might increase acceptance
    score_factor = (sample_acceptance_df['match_score'] - 0.5) * 2
    salary_factor = sample_acceptance_df['salary_diff_percentage'] / 20
    remote_factor = (sample_acceptance_df['is_remote'] - 0.5) * 0.5
    combined_logit = score_factor + salary_factor + remote_factor - 0.2 # Base tendency
    prob = 1 / (1 + np.exp(-combined_logit))
    sample_acceptance_df['accepted_offer'] = (prob > np.random.rand(50)).astype(int)
    
    # Ensure there are enough samples for each class after generation
    if sample_acceptance_df['accepted_offer'].nunique() < 2:
        print("Warning: Dummy data generation resulted in only one class for 'accepted_offer'. Adjusting...")
        # Flip some labels to ensure two classes for demonstration
        num_to_flip = max(1, len(sample_acceptance_df) // 10)
        flip_indices = np.random.choice(sample_acceptance_df.index, num_to_flip, replace=False)
        sample_acceptance_df.loc[flip_indices, 'accepted_offer'] = 1 - sample_acceptance_df.loc[flip_indices, 'accepted_offer']

    print(f"Value counts for 'accepted_offer' in dummy data:\n{sample_acceptance_df['accepted_offer'].value_counts()}")

    print("\n--- Training Acceptance Predictor ---")
    trained_pipeline, metrics = train_acceptance_predictor(sample_acceptance_df.copy(), target_column='accepted_offer')

    if trained_pipeline:
        print(f"\nTraining/Test Metrics: ROC AUC = {metrics.get('test_roc_auc', 'N/A'):.4f}")
        print("\n--- Predicting Acceptance Probability for New Candidates ---")
        new_candidate_features = pd.DataFrame({
            'match_score': [0.85, 0.60, 0.92, 0.50],
            'salary_diff_percentage': [5, -10, 15, 0],
            'candidate_experience_years': [5, 2, 10, 3],
            'is_remote': [1, 0, 1, 1]
        })
        
        acceptance_probs = predict_acceptance_probability(new_candidate_features.copy(), model_pipeline=trained_pipeline)
        if acceptance_probs is not None:
            for i, prob in enumerate(acceptance_probs):
                print(f"Candidate {i+1} features -> Predicted Acceptance Probability: {prob:.2%}")

        print("\n--- Predicting using the saved model (if available) ---")
        if os.path.exists(ACCEPTANCE_MODEL_SAVE_PATH):
            acceptance_probs_saved = predict_acceptance_probability(new_candidate_features.copy()) # Loads model
            if acceptance_probs_saved is not None:
                print("Predictions from saved model:")
                for i, prob in enumerate(acceptance_probs_saved):
                    print(f"Candidate {i+1} features -> Predicted Acceptance Probability: {prob:.2%}")
        else:
            print(f"Saved model {ACCEPTANCE_MODEL_SAVE_PATH} not found. Skipping this test.")
    else:
        print("Skipping prediction as acceptance model training failed.")

    print("\nCandidate Acceptance Prediction Module execution finished.")
    # Notes:
    # - Feature engineering is key: 
    #   - 'salary_diff_percentage' (offered vs. market/expected) is a strong predictor.
    #   - Candidate's current situation (e.g., actively looking, employed).
    #   - Company reputation, benefits, interview experience (harder to quantify).
    #   - Time to offer, number of other offers.
    # - Requires historical data of offers made and their outcomes (accepted/rejected).