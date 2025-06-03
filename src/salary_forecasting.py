# src/salary_forecasting.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor #, RandomForestRegressor
import xgboost as xgb # For XGBoost
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# --- Configuration ---
SALARY_MODEL_SAVE_PATH = "models/salary_forecaster_xgb.joblib"

# --- Feature Engineering (Conceptual) ---
# In a real scenario, you'd extract features like:
# - Years of experience (from resume text or structured data)
# - Specific skills (one-hot encoded or using embeddings)
# - Location (one-hot encoded or using geo-data)
# - Job Title (standardized and encoded)
# - Company size/type (if available)
# - Education level

# --- Model Training Function ---
def train_salary_forecaster(job_features_df, target_salary_column='salary_midpoint', output_model_path=None):
    """
    Trains a regression model to predict salary.

    Args:
        job_features_df (pd.DataFrame): DataFrame with features and target salary.
                                        Example features: 'skills_list', 'location', 'experience_level_numeric'.
        target_salary_column (str): Name of the column containing the numerical salary target.
        output_model_path (str, optional): Path to save the trained model pipeline.
                                           Defaults to SALARY_MODEL_SAVE_PATH.

    Returns:
        sklearn.pipeline.Pipeline: The trained regression pipeline (preprocessor + model).
        dict: Training and test metrics.
    """
    if not isinstance(job_features_df, pd.DataFrame) or target_salary_column not in job_features_df.columns:
        print("Error: Invalid input DataFrame or target column name.")
        return None, None

    # Define feature types for preprocessing
    # This is highly dependent on the actual features available.
    # For this example, let's assume we have 'location' (categorical) and 'experience_numeric' (numerical)
    # and 'skills_flat_text' (text to be vectorized, or use pre-computed skill counts/embeddings)
    
    # For simplicity, we'll assume 'skills_flat_text' is a string of concatenated skills for TF-IDF
    # and 'location' is categorical, 'experience_numeric' is numerical.
    categorical_features = ['location'] # Example
    numerical_features = ['experience_numeric'] # Example
    # text_feature = 'skills_flat_text' # If using TF-IDF for skills directly in pipeline

    # Ensure all necessary columns exist
    required_cols = categorical_features + numerical_features + [target_salary_column]
    # if text_feature: required_cols.append(text_feature)
    
    missing_cols = [col for col in required_cols if col not in job_features_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in DataFrame: {missing_cols}")
        return None, None

    # Handle NaNs
    # For simplicity, drop rows with NaN in target or key features. More sophisticated imputation can be used.
    job_features_df = job_features_df.dropna(subset=required_cols)
    if job_features_df.empty:
        print("Error: DataFrame is empty after dropping NaNs.")
        return None, None

    X = job_features_df.drop(columns=[target_salary_column])
    y = job_features_df[target_salary_column]

    if len(X) < 10: # Arbitrary small number, ensure enough data
        print("Error: Not enough data points to train the model effectively.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create preprocessing pipelines for different feature types
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for dense array
    # text_transformer = TfidfVectorizer(stop_words='english', max_features=100) # If using TF-IDF for skills

    # Bundle preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            # ('txt', text_transformer, text_feature) # If using TF-IDF for skills
        ],
        remainder='passthrough' # Keep other columns not specified (if any)
    )

    # Define the model (XGBoost Regressor is a good choice)
    # model = LinearRegression()
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, 
                             max_depth=5, random_state=42, early_stopping_rounds=10)
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    print("Training the salary forecaster...")
    # For XGBoost with early stopping, we need to pass eval_set to fit method of the regressor step
    # This requires a bit more care if preprocessor transforms X_test differently or if feature names change.
    # A simpler way for now is to fit without early stopping directly in pipeline.fit
    # Or, transform data first, then fit model with eval_set.
    
    # Fit the pipeline (preprocessor will be fit on X_train and transform X_train)
    # Then regressor will be fit on the transformed X_train.
    # For XGBoost early stopping in a pipeline, we need to prepare the eval_set
    # by transforming X_test with the preprocessor fitted on X_train.
    
    # Create a temporary pipeline with just the preprocessor to transform X_test for eval_set
    temp_preprocessor_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    temp_preprocessor_pipeline.fit(X_train) # Fit preprocessor on X_train
    X_test_transformed_for_eval = temp_preprocessor_pipeline.transform(X_test)

    fit_params = {
        'regressor__eval_set': [(X_test_transformed_for_eval, y_test)],
        'regressor__verbose': False # Suppress XGBoost verbosity during training with early stopping
        # 'regressor__early_stopping_rounds': 10 # This is already in constructor, but can be overridden here
    }
    
    pipeline.fit(X_train, y_train, **fit_params)
    
    # Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"Train RMSE: {train_rmse:.2f}, Train R2: {train_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f}, Test R2: {test_r2:.4f}, Test MAE: {test_mae:.2f}")

    metrics = {
        "train_rmse": train_rmse, "train_r2": train_r2,
        "test_rmse": test_rmse, "test_r2": test_r2, "test_mae": test_mae
    }

    # Determine save path
    save_path = output_model_path if output_model_path else SALARY_MODEL_SAVE_PATH
    
    # Save the model pipeline
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"Trained salary forecasting pipeline saved to {save_path}")

    return pipeline, metrics

# --- Prediction Function ---
def predict_salary(job_features_df, model_pipeline=None, input_model_path=None):
    """
    Predicts salary for new job feature sets.

    Args:
        job_features_df (pd.DataFrame): DataFrame with features for prediction.
                                        Must have the same columns as used in training (before preprocessing).
        model_pipeline (sklearn.pipeline.Pipeline, optional): Trained model pipeline.
                                                              If None, attempts to load from path.
        input_model_path (str, optional): Path to load the trained model pipeline from.
                                          Defaults to SALARY_MODEL_SAVE_PATH if model_pipeline is None.

    Returns:
        np.ndarray: Predicted salaries.
    """
    if model_pipeline is None:
        load_path = input_model_path if input_model_path else SALARY_MODEL_SAVE_PATH
        try:
            model_pipeline = joblib.load(load_path)
            print(f"Loaded salary forecasting pipeline from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Train first.")
            return None
        except Exception as e:
            print(f"Error loading model from {load_path}: {e}")
            return None

    if not isinstance(job_features_df, pd.DataFrame) or job_features_df.empty:
        print("Error: Input for prediction must be a non-empty DataFrame.")
        return None

    # Ensure all required columns are present (matching training)
    # This check is implicitly handled by the preprocessor in the pipeline if columns are missing.
    # However, it's good practice to ensure the input schema is as expected.

    try:
        predictions = model_pipeline.predict(job_features_df)
        return predictions
    except Exception as e:
        print(f"Error during salary prediction: {e}")
        return None

# --- Demand Forecasting (Conceptual) ---
# Demand forecasting is more complex and would typically involve:
# - Time series analysis of job postings (e.g., ARIMA, Prophet).
# - Analyzing trends in skills, locations, job titles over time.
# - Incorporating economic indicators.
# This is beyond a simple regression model for salary.

def analyze_demand_trends(historical_job_data):
    """Conceptual placeholder for demand trend analysis."""
    print("\n--- Demand Trend Analysis (Conceptual) ---")
    if not isinstance(historical_job_data, pd.DataFrame) or 'date_posted' not in historical_job_data.columns:
        print("Requires historical job data with a 'date_posted' column.")
        return None

    # Example: Count job postings over time for a specific role or skill
    # historical_job_data['date_posted'] = pd.to_datetime(historical_job_data['date_posted'])
    # monthly_counts = historical_job_data.resample('M', on='date_posted').size()
    # print("Monthly job postings trend (example):")
    # print(monthly_counts)
    print("Demand forecasting would involve time series models and analysis of job market dynamics.")
    print("This is a placeholder for a more complex module.")
    return None # Placeholder

# --- Sample Usage ---
if __name__ == "__main__":
    print("Starting Salary Forecasting Module...")

    # Sample data (highly simplified)
    # In reality, 'experience_numeric' would be extracted/standardized (e.g., from years or levels)
    # 'skills_flat_text' would be a concatenation of skills for TF-IDF, or you'd use skill counts/embeddings.
    # 'salary_midpoint' is the target variable.
    data = {
        'job_id': [f'job{i}' for i in range(1, 21)],
        'location': ['New York', 'San Francisco', 'Austin', 'Chicago', 'New York', 
                     'San Francisco', 'Austin', 'Remote', 'New York', 'San Francisco', 
                     'Chicago', 'Remote', 'Austin', 'New York', 'San Francisco', 
                     'Austin', 'Chicago', 'Remote', 'New York', 'San Francisco'],
        'experience_numeric': [2, 5, 3, 7, 1, 6, 4, 8, 3, 9, 2, 5, 6, 4, 7, 2, 8, 3, 6, 10], # e.g., years
        # 'skills_flat_text': [ # Example if using TF-IDF for skills in pipeline
        #     'python java sql agile', 'react nodejs javascript aws', 'machine learning python statistics', ...
        # ],
        'salary_midpoint': [90000, 150000, 110000, 160000, 75000, 
                            170000, 120000, 180000, 100000, 200000, 
                            95000, 140000, 130000, 115000, 190000, 
                            105000, 175000, 125000, 155000, 220000]
    }
    sample_salary_df = pd.DataFrame(data)

    print("\n--- Training Salary Forecaster ---")
    trained_pipeline, metrics = train_salary_forecaster(sample_salary_df.copy(), target_salary_column='salary_midpoint')

    if trained_pipeline:
        print(f"\nTraining/Test Metrics: {metrics}")
        print("\n--- Predicting Salaries for New Data ---")
        new_job_features = pd.DataFrame({
            'location': ['New York', 'Remote', 'San Francisco'],
            'experience_numeric': [3, 5, 8],
            # 'skills_flat_text': ['python machine learning cloud', 'java spring agile', 'react javascript fullstack']
        })
        
        predicted_salaries = predict_salary(new_job_features.copy(), model_pipeline=trained_pipeline)
        if predicted_salaries is not None:
            for i, row in new_job_features.iterrows():
                print(f"Predicted salary for features (Loc: {row['location']}, Exp: {row['experience_numeric']}): ${predicted_salaries[i]:.0f}")

        print("\n--- Predicting using the saved model (if available) ---")
        if os.path.exists(SALARY_MODEL_SAVE_PATH):
            predicted_salaries_from_saved = predict_salary(new_job_features.copy()) # Loads model
            if predicted_salaries_from_saved is not None:
                print("Predictions from saved model:")
                for i, row in new_job_features.iterrows():
                    print(f"Predicted salary (Loc: {row['location']}, Exp: {row['experience_numeric']}): ${predicted_salaries_from_saved[i]:.0f}")
        else:
            print(f"Saved model {SALARY_MODEL_SAVE_PATH} not found. Skipping this test.")
    else:
        print("Skipping prediction as salary model training failed.")

    # Conceptual call to demand forecasting
    # analyze_demand_trends(sample_salary_df.assign(date_posted=pd.to_datetime('today'))) # Dummy date

    print("\nSalary Forecasting Module execution finished.")
    # Notes:
    # - Feature engineering is CRITICAL for salary prediction. More granular skills, 
    #   standardized experience levels, and detailed location data (city, state, country) improve accuracy.
    # - Consider using log-transformed salary if its distribution is skewed.
    # - Hyperparameter tuning for the regressor (e.g., XGBoost) is important.
    # - Demand forecasting is a separate, complex task usually involving time series analysis.