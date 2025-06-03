# src/evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Classification Metrics ---
def calculate_classification_metrics(y_true, y_pred, average='weighted'):
    """
    Calculates and returns common classification metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        average (str, optional): Averaging method for precision, recall, F1-score 
                                 (e.g., 'binary', 'micro', 'macro', 'weighted', 'samples'). 
                                 Defaults to 'weighted'.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
              Returns None if inputs are invalid.
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        print("Error: y_true and y_pred must have the same non-zero length.")
        return None
    
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return metrics
    except Exception as e:
        print(f"Error calculating classification metrics: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix'):
    """
    Plots a confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list, optional): List of class names for axis labels. Defaults to None.
        title (str, optional): Title for the plot. Defaults to 'Confusion Matrix'.
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        print("Error: y_true and y_pred must have the same non-zero length for confusion matrix.")
        return

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else 'auto', 
                yticklabels=class_names if class_names else 'auto')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# --- 2. Regression Metrics ---
def calculate_regression_metrics(y_true, y_pred):
    """
    Calculates common regression metrics.

    Args:
        y_true (array-like): True continuous values.
        y_pred (array-like): Predicted continuous values.

    Returns:
        dict: A dictionary containing Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
              Returns None if inputs are invalid.
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        print("Error: y_true and y_pred must have the same non-zero length for regression metrics.")
        return None

    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'rmse': rmse
        }
        return metrics
    except Exception as e:
        print(f"Error calculating regression metrics: {e}")
        return None

# --- 3. Recommendation System Metrics (Conceptual) ---
def evaluate_recommendations(recommendations_dict, actual_interactions_dict, k_values=None):
    """
    Conceptual function to evaluate recommendation system performance.
    Common metrics include Precision@k, Recall@k, MAP@k, nDCG@k.

    Args:
        recommendations_dict (dict): 
            Keys are user_ids, values are lists of recommended item_ids.
            Example: {'user1': ['itemA', 'itemB'], 'user2': ['itemC']}
        actual_interactions_dict (dict): 
            Keys are user_ids, values are lists of actually interacted/liked item_ids.
            Example: {'user1': ['itemA', 'itemD'], 'user2': ['itemC', 'itemE']}
        k_values (list of int, optional): 
            List of 'k' values for which to calculate Precision@k and Recall@k. 
            Defaults to [5, 10].

    Returns:
        dict: A dictionary containing average Precision@k and Recall@k for specified k values.
    """
    if k_values is None:
        k_values = [5, 10]

    results = {}
    for k in k_values:
        precisions_at_k = []
        recalls_at_k = []
        
        for user_id, recommended_items in recommendations_dict.items():
            if user_id not in actual_interactions_dict:
                continue # User has no actual interactions to compare against
            
            actual_items = set(actual_interactions_dict[user_id])
            if not actual_items: # No positive interactions for this user
                precisions_at_k.append(0) # Or handle as per specific evaluation protocol
                recalls_at_k.append(0)
                continue

            top_k_recommended = recommended_items[:k]
            relevant_and_recommended = [item for item in top_k_recommended if item in actual_items]
            
            # Precision@k = (Number of relevant items in top-k) / k
            precision_k = len(relevant_and_recommended) / k if k > 0 else 0
            precisions_at_k.append(precision_k)
            
            # Recall@k = (Number of relevant items in top-k) / (Total number of relevant items)
            recall_k = len(relevant_and_recommended) / len(actual_items) if len(actual_items) > 0 else 0
            recalls_at_k.append(recall_k)
        
        results[f'avg_precision_at_{k}'] = np.mean(precisions_at_k) if precisions_at_k else 0
        results[f'avg_recall_at_{k}'] = np.mean(recalls_at_k) if recalls_at_k else 0
        
    print("\n--- Recommendation System Evaluation (Conceptual) ---")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    return results

# --- Sample Usage ---
if __name__ == "__main__":
    print("Starting Evaluation Module...")

    # --- Classification Example ---
    print("\n--- Classification Metrics Example ---")
    y_true_clf = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    y_pred_clf = [0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
    class_names_clf = ['Class 0', 'Class 1']

    clf_metrics = calculate_classification_metrics(y_true_clf, y_pred_clf, average='binary') # for binary case
    if clf_metrics:
        print("Binary Classification Metrics:")
        for metric, value in clf_metrics.items():
            print(f"  {metric}: {value:.4f}")
        plot_confusion_matrix(y_true_clf, y_pred_clf, class_names=class_names_clf, title='Binary Classification CM')

    # Multi-class example
    y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred_multi = [0, 1, 1, 0, 2, 2, 0, 0, 2]
    class_names_multi = ['Category A', 'Category B', 'Category C']
    multi_clf_metrics = calculate_classification_metrics(y_true_multi, y_pred_multi, average='weighted')
    if multi_clf_metrics:
        print("\nMulti-class Classification Metrics (Weighted Average):")
        for metric, value in multi_clf_metrics.items():
            print(f"  {metric}: {value:.4f}")
        plot_confusion_matrix(y_true_multi, y_pred_multi, class_names=class_names_multi, title='Multi-class Classification CM')

    # --- Regression Example ---
    print("\n--- Regression Metrics Example ---")
    y_true_reg = [10.0, 12.5, 15.0, 18.2, 20.5, 22.0]
    y_pred_reg = [10.5, 12.0, 14.5, 19.0, 20.0, 23.0]
    
    reg_metrics = calculate_regression_metrics(y_true_reg, y_pred_reg)
    if reg_metrics:
        print("Regression Metrics:")
        for metric, value in reg_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # --- Recommendation System Evaluation Example (Conceptual) ---
    sample_recommendations = {
        'user1': ['jobA', 'jobB', 'jobC', 'jobD', 'jobE'],
        'user2': ['jobF', 'jobG', 'jobH', 'jobI', 'jobJ'],
        'user3': ['jobA', 'jobK', 'jobL', 'jobM', 'jobN']
    }
    sample_actual_interactions = {
        'user1': ['jobA', 'jobC', 'jobX'], # Liked jobA, jobC
        'user2': ['jobG', 'jobY', 'jobZ', 'jobI'], # Liked jobG, jobI
        'user3': ['jobO', 'jobP'] # Liked none of the recommended ones
    }
    
    rec_eval_metrics = evaluate_recommendations(sample_recommendations, sample_actual_interactions, k_values=[3, 5])
    # The results are printed within the function for this example.

    print("\nEvaluation Module execution finished.")
    # Note: For real-world scenarios, ensure that the 'average' parameter for classification metrics
    # is chosen appropriately based on the problem (binary, multiclass, imbalanced data etc.).
    # Recommendation evaluation can be much more complex, involving offline and online (A/B testing) methods.