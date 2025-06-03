# src/job_classification.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib # For saving and loading the model
import os

# --- Configuration ---
MODEL_SAVE_PATH = "models/job_classifier_dt.joblib"

# --- Model Training Function ---
def train_job_classifier(job_data, text_column='cleaned_description', label_column='job_category', output_model_path=None):
    """
    Trains a Decision Tree classifier to categorize job roles.

    Args:
        job_data (pd.DataFrame): DataFrame containing job descriptions and their categories.
        text_column (str): Name of the column with preprocessed job description text.
        label_column (str): Name of the column with job category labels.
        output_model_path (str, optional): Path to save the trained model pipeline.
                                           Defaults to MODEL_SAVE_PATH.

    Returns:
        sklearn.pipeline.Pipeline: The trained classification pipeline (vectorizer + classifier).
        dict: A dictionary containing training metrics (e.g., accuracy, classification report).
    """
    if not isinstance(job_data, pd.DataFrame) or text_column not in job_data.columns or label_column not in job_data.columns:
        print("Error: Invalid input data or column names.")
        return None, None
    
    if job_data[text_column].isnull().any() or job_data[label_column].isnull().any():
        print("Warning: Data contains null values. Attempting to drop them.")
        job_data = job_data.dropna(subset=[text_column, label_column])
        if job_data.empty:
            print("Error: No data left after dropping nulls.")
            return None, None

    X = job_data[text_column]
    y = job_data[label_column]

    if len(y.unique()) < 2:
        print("Error: Need at least two distinct classes for classification.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None)

    # Create a pipeline: TF-IDF Vectorizer -> Decision Tree Classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1,2))),
        ('clf', DecisionTreeClassifier(random_state=42, class_weight='balanced')) # Added class_weight for imbalanced datasets
    ])

    print("Training the job classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report (Test Set):")
    report_str = classification_report(y_test, y_pred_test, zero_division=0)
    print(report_str)

    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "classification_report_test": classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    }
    
    # Determine save path
    save_path = output_model_path if output_model_path else MODEL_SAVE_PATH
    
    # Save the trained pipeline (model + vectorizer)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"Trained model pipeline saved to {save_path}")
    
    # Note: For TF-IDF, the vectorizer is part of the pipeline. 
    # If using embeddings directly, you might not need a separate vectorizer here.

    return pipeline, metrics

# --- Prediction Function ---
def predict_job_category(job_descriptions_list, model_pipeline=None, input_model_path=None):
    """
    Predicts job categories for a list of job descriptions using a trained model.

    Args:
        job_descriptions_list (list of str): A list of preprocessed job description texts.
        model_pipeline (sklearn.pipeline.Pipeline, optional): The trained model pipeline.
                                                              If None, attempts to load model from path.
        input_model_path (str, optional): Path to load the trained model pipeline from.
                                          Defaults to MODEL_SAVE_PATH if model_pipeline is None.

    Returns:
        list: A list of predicted job categories.
    """
    if model_pipeline is None:
        load_path = input_model_path if input_model_path else MODEL_SAVE_PATH
        try:
            model_pipeline = joblib.load(load_path)
            print(f"Loaded model pipeline from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Please train the model first.")
            return None
        except Exception as e:
            print(f"Error loading model from {load_path}: {e}")
            return None

    if not isinstance(job_descriptions_list, list) or not all(isinstance(desc, str) for desc in job_descriptions_list):
        print("Error: Input must be a list of strings.")
        return None
    
    if not job_descriptions_list:
        print("Input job descriptions list is empty.")
        return []

    try:
        predictions = model_pipeline.predict(job_descriptions_list)
        return predictions.tolist()
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# --- Sample Usage ---
if __name__ == "__main__":
    print("Starting Job Classification Module...")

    # Sample Data (replace with actual data loading and preprocessing)
    # In a real scenario, 'cleaned_description' would come from the preprocessing module.
    data = {
        'cleaned_description': [
            'software engineer python java agile development backend systems', # Software Engineer
            'develop web applications using react nodejs javascript frontend', # Web Developer
            'data scientist machine learning statistics python modeling analysis', # Data Scientist
            'senior software engineer leadership python microservices cloud', # Software Engineer
            'ui ux designer figma sketch user research interactive design', # UI/UX Designer
            'data analyst sql tableau reporting insights business intelligence', # Data Analyst
            'full stack developer javascript python react django database', # Web Developer
            'machine learning engineer nlp deep learning tensorflow pytorch', # Data Scientist
            'lead software architect design patterns system design scalable solutions', # Software Engineer (or Architect)
            'product manager agile roadmap user stories market research product lifecycle' # Product Manager
        ],
        'job_category': [
            'Software Engineer',
            'Web Developer',
            'Data Scientist',
            'Software Engineer',
            'UI/UX Designer',
            'Data Analyst',
            'Web Developer',
            'Data Scientist',
            'Software Engineer', # Could also be 'Software Architect'
            'Product Manager'
        ]
    }
    sample_df = pd.DataFrame(data)

    # Add more data for better training and to ensure all categories have enough samples
    more_data = {
        'cleaned_description': [
            'java developer spring boot microservices api design enterprise applications', # Software Engineer
            'frontend developer html css javascript responsive design user interface', # Web Developer
            'research scientist deep learning computer vision publications phd', # Data Scientist (or Researcher)
            'devops engineer aws kubernetes docker ci cd automation infrastructure', # DevOps Engineer
            'graphic designer adobe creative suite branding illustration digital art', # UI/UX Designer (or Graphic Designer)
            'business analyst requirements gathering process improvement stakeholder management', # Data Analyst (or Business Analyst)
            'mobile developer android kotlin ios swift native applications', # Mobile Developer
            'ai researcher reinforcement learning robotics publications academic', # Data Scientist (or Researcher)
            'technical lead mentoring code reviews project delivery agile practices', # Software Engineer
            'marketing manager digital marketing seo content strategy social media campaigns' # Marketing Manager
        ],
        'job_category': [
            'Software Engineer',
            'Web Developer',
            'Data Scientist', 
            'DevOps Engineer',
            'UI/UX Designer',
            'Data Analyst',
            'Mobile Developer',
            'Data Scientist',
            'Software Engineer',
            'Marketing Manager'
        ]
    }
    sample_df = pd.concat([sample_df, pd.DataFrame(more_data)], ignore_index=True)
    
    print("\n--- Training Job Classifier ---")
    # Ensure there are enough samples per class for stratification
    # For this example, we'll just proceed. In practice, handle small classes carefully.
    trained_model, training_metrics = train_job_classifier(sample_df, 'cleaned_description', 'job_category')

    if trained_model:
        print("\n--- Predicting Job Categories for New Data ---")
        new_job_descs_text = [
            'looking for a python developer with experience in django and rest apis', # Expected: Software Engineer or Web Developer
            'data analysis and visualization expert needed power bi sql proficiency', # Expected: Data Analyst
            'creative ui designer for mobile apps must know figma and user testing', # Expected: UI/UX Designer
            'entry level java programmer for backend development tasks', # Expected: Software Engineer
            'artificial intelligence engineer working on cutting edge nlp models' # Expected: Data Scientist
        ]
        
        predictions = predict_job_category(new_job_descs_text, model_pipeline=trained_model)
        if predictions:
            for desc, category in zip(new_job_descs_text, predictions):
                print(f"Description: '{desc[:50]}...' -> Predicted Category: {category}")

        print("\n--- Predicting using the saved model (if available) ---")
        # This part simulates loading the model in a different session/script
        if os.path.exists(MODEL_SAVE_PATH):
            predictions_from_saved_model = predict_job_category(new_job_descs_text) # model_pipeline is None, so it will load
            if predictions_from_saved_model:
                print("Predictions from saved model:")
                for desc, category in zip(new_job_descs_text, predictions_from_saved_model):
                    print(f"Description: '{desc[:50]}...' -> Predicted Category: {category}")
        else:
            print(f"Saved model {MODEL_SAVE_PATH} not found. Skipping this test.")
    else:
        print("Skipping prediction as model training failed or was not performed.")

    print("\nJob Classification Module execution finished.")

    # Note: 
    # 1. The quality of classification heavily depends on the quality and quantity of training data 
    #    and the preprocessing steps.
    # 2. For more complex scenarios, consider using embeddings as features instead of TF-IDF, 
    #    or more advanced classifiers (e.g., SVM, RandomForest, GradientBoosting, or Neural Networks).
    # 3. Hyperparameter tuning (e.g., using GridSearchCV) for both TfidfVectorizer and 
    #    DecisionTreeClassifier would improve performance.