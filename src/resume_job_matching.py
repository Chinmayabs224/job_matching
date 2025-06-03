# src/resume_job_matching.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC # For SVM-based matching (placeholder)
# from tensorflow.keras.models import Model # For ANN-based matching (placeholder)
import joblib
import os

# Assuming embedding.py and preprocessing.py are in the same directory or accessible
# For direct execution, you might need to adjust sys.path or ensure modules are installed
try:
    from .embedding import get_text_embeddings, EMBEDDING_MODEL, load_embedding_model
    from .preprocessing import extract_skills_keywords # For keyword overlap
except ImportError:
    # Fallback for direct execution (if not run as part of a package)
    print("Attempting to import from parent directory for resume_job_matching.py")
    from embedding import get_text_embeddings, EMBEDDING_MODEL, load_embedding_model
    from preprocessing import extract_skills_keywords

# --- Configuration ---
SVM_MATCHER_MODEL_PATH = "models/svm_matcher.joblib"

# --- 1. Keyword Overlap Matching ---
def calculate_keyword_overlap(resume_text, job_desc_text, resume_skills=None, job_skills=None):
    """
    Calculates a simple keyword overlap score based on extracted skills.
    Args:
        resume_text (str): Preprocessed text of the resume.
        job_desc_text (str): Preprocessed text of the job description.
        resume_skills (list, optional): Pre-extracted skills from the resume.
        job_skills (list, optional): Pre-extracted skills from the job description.

    Returns:
        float: A score between 0 and 1 representing the Jaccard similarity of skills.
               Returns 0.0 if no skills are found or inputs are invalid.
    """
    if resume_skills is None:
        # print("Extracting skills from resume for keyword overlap...")
        resume_skills = extract_skills_keywords(resume_text)
    if job_skills is None:
        # print("Extracting skills from job description for keyword overlap...")
        job_skills = extract_skills_keywords(job_desc_text)

    set_resume_skills = set(s.lower() for s in resume_skills if isinstance(s, str))
    set_job_skills = set(s.lower() for s in job_skills if isinstance(s, str))

    if not set_resume_skills or not set_job_skills:
        return 0.0 # No common basis for comparison

    intersection = len(set_resume_skills.intersection(set_job_skills))
    union = len(set_resume_skills.union(set_job_skills))
    
    return intersection / union if union > 0 else 0.0

# --- 2. Cosine Similarity Matching (using Text Embeddings) ---
def calculate_cosine_similarity_score(resume_embedding, job_embedding):
    """
    Calculates cosine similarity between a resume embedding and a job embedding.
    Args:
        resume_embedding (np.ndarray): Embedding vector for the resume.
        job_embedding (np.ndarray): Embedding vector for the job description.

    Returns:
        float: Cosine similarity score (between -1 and 1, typically 0 to 1 for SBERT).
               Returns 0.0 if embeddings are invalid.
    """
    if resume_embedding is None or job_embedding is None or resume_embedding.ndim == 0 or job_embedding.ndim == 0:
        # print("Warning: Invalid embeddings provided for cosine similarity.")
        return 0.0
    
    # Ensure embeddings are 2D arrays for cosine_similarity function
    if resume_embedding.ndim == 1:
        resume_embedding = resume_embedding.reshape(1, -1)
    if job_embedding.ndim == 1:
        job_embedding = job_embedding.reshape(1, -1)
        
    if resume_embedding.shape[1] != job_embedding.shape[1]:
        # print(f"Warning: Embedding dimensions mismatch: Resume {resume_embedding.shape}, Job {job_embedding.shape}")
        return 0.0

    similarity = cosine_similarity(resume_embedding, job_embedding)
    return similarity[0][0]

# --- 3. Intelligent Matching (SVM/ANN - Placeholder) ---
# This requires a trained model. For now, it's a placeholder.
# Training data would consist of (resume_features, job_features, match_label [0 or 1])

def train_svm_matcher(features, labels):
    """
    Placeholder for training an SVM-based matching model.
    Features could be concatenated embeddings, difference of embeddings, or other engineered features.
    """
    print("Training SVM matcher (placeholder)...")
    model = SVC(probability=True, random_state=42) # probability=True for predict_proba
    # model.fit(features, labels) # This would be actual training data
    # os.makedirs(os.path.dirname(SVM_MATCHER_MODEL_PATH), exist_ok=True)
    # joblib.dump(model, SVM_MATCHER_MODEL_PATH)
    # print(f"SVM matcher model saved to {SVM_MATCHER_MODEL_PATH}")
    print("SVM training is a placeholder. No actual model trained or saved.")
    return model # In reality, this would be the trained model

def predict_svm_match_score(resume_features, job_features, model=None):
    """
    Placeholder for predicting match score using a trained SVM model.
    Returns a pseudo-score (e.g., based on cosine similarity for now).
    """
    # if model is None:
    #     try:
    #         model = joblib.load(SVM_MATCHER_MODEL_PATH)
    #     except FileNotFoundError:
    #         print(f"SVM model not found at {SVM_MATCHER_MODEL_PATH}. Returning 0.")
    #         return 0.0
    
    # combined_features = np.concatenate((resume_features, job_features)) # Example feature engineering
    # score = model.predict_proba(combined_features.reshape(1, -1))[0][1] # Probability of class 1 (match)
    
    print("SVM match prediction is a placeholder. Using cosine similarity as a proxy.")
    if isinstance(resume_features, np.ndarray) and isinstance(job_features, np.ndarray):
        return calculate_cosine_similarity_score(resume_features, job_features) # Placeholder behavior
    return 0.0

# --- Combined Matching Score ---
def get_combined_match_score(resume_text, job_desc_text, resume_embedding, job_embedding,
                             resume_skills_list=None, job_skills_list=None,
                             weights={"keyword": 0.2, "cosine": 0.5, "svm": 0.3}):
    """
    Calculates a weighted combined match score.
    Args:
        resume_text (str): Preprocessed text of the resume.
        job_desc_text (str): Preprocessed text of the job description.
        resume_embedding (np.ndarray): Embedding for the resume.
        job_embedding (np.ndarray): Embedding for the job description.
        resume_skills_list (list, optional): Pre-extracted skills from resume.
        job_skills_list (list, optional): Pre-extracted skills from job.
        weights (dict): Weights for combining the scores.

    Returns:
        float: Combined match score (0 to 1).
        dict: Dictionary of individual scores.
    """
    keyword_score = calculate_keyword_overlap(resume_text, job_desc_text, resume_skills_list, job_skills_list)
    cosine_score = calculate_cosine_similarity_score(resume_embedding, job_embedding)
    
    # For SVM, we'd use features derived from embeddings or text
    # As a placeholder, we'll use the embeddings directly for the SVM proxy
    svm_score = predict_svm_match_score(resume_embedding, job_embedding) # Placeholder

    total_score = (
        weights["keyword"] * keyword_score +
        weights["cosine"] * cosine_score +
        weights["svm"] * svm_score
    )
    
    individual_scores = {
        "keyword_overlap": keyword_score,
        "cosine_similarity": cosine_score,
        "svm_intelligent_match (proxy)": svm_score
    }
    
    return total_score, individual_scores

# --- Main function to match one resume to multiple jobs ---
def match_resume_to_jobs(resume_text_processed, resume_embedding_vector, 
                         jobs_processed_data, job_embeddings_matrix,
                         resume_skills=None):
    """
    Matches a single resume against a list of job descriptions.

    Args:
        resume_text_processed (str): Preprocessed text of the resume.
        resume_embedding_vector (np.ndarray): Embedding of the resume.
        jobs_processed_data (list of dict): List of dicts, each containing preprocessed job data 
                                           (e.g., 'cleaned_text', 'skills', 'job_id').
        job_embeddings_matrix (np.ndarray): Matrix of embeddings for all job descriptions.
        resume_skills (list, optional): Pre-extracted skills from the resume.

    Returns:
        list: A list of tuples (job_id, combined_score, individual_scores_dict), sorted by combined_score descending.
    """
    if EMBEDDING_MODEL is None:
        print("Embedding model not loaded. Attempting to load...")
        load_embedding_model() # Try to load the default embedding model
        if EMBEDDING_MODEL is None:
            print("Failed to load embedding model. Cannot perform matching.")
            return []

    if resume_embedding_vector is None or resume_embedding_vector.ndim == 0:
        print("Invalid resume embedding. Cannot perform matching.")
        return []
    if job_embeddings_matrix is None or job_embeddings_matrix.ndim != 2 or job_embeddings_matrix.shape[0] != len(jobs_processed_data):
        print("Invalid job embeddings matrix or mismatch with job data. Cannot perform matching.")
        return []

    matches = []
    for i, job_data in enumerate(jobs_processed_data):
        job_text_processed = job_data.get('cleaned_text', '')
        job_embedding_vector = job_embeddings_matrix[i]
        job_skills = job_data.get('skills', [])
        job_id = job_data.get('job_id', f'job_{i}') # Use a default ID if not present

        combined_score, individual_scores = get_combined_match_score(
            resume_text_processed, 
            job_text_processed, 
            resume_embedding_vector, 
            job_embedding_vector,
            resume_skills_list=resume_skills,
            job_skills_list=job_skills
        )
        matches.append((job_id, combined_score, individual_scores))

    # Sort matches by combined score in descending order
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

# --- Sample Usage ---
if __name__ == "__main__":
    print("Starting Resume-Job Matching Module...")

    # Ensure embedding model is loaded (it's also called in get_text_embeddings if not loaded)
    if EMBEDDING_MODEL is None:
        load_embedding_model()

    if EMBEDDING_MODEL:
        # Sample preprocessed data (normally from preprocessing.py and embedding.py)
        sample_resume_text_p = "experienced software engineer python java machine learning projects agile"
        sample_resume_skills = ["python", "java", "machine learning", "agile"]
        
        sample_jobs_data_p = [
            {'job_id': 'job1', 'cleaned_text': 'senior python developer aws cloud microservices', 'skills': ['python', 'aws', 'cloud', 'microservices']},
            {'job_id': 'job2', 'cleaned_text': 'java engineer spring boot backend development sql', 'skills': ['java', 'spring boot', 'sql', 'backend']},
            {'job_id': 'job3', 'cleaned_text': 'machine learning specialist nlp tensorflow research python', 'skills': ['machine learning', 'nlp', 'tensorflow', 'python', 'research']},
            {'job_id': 'job4', 'cleaned_text': 'entry level software developer c++ problem solving data structures', 'skills': ['c++', 'problem solving', 'data structures']}
        ]

        # Generate embeddings (mocking what embedding.py would do)
        print("\nGenerating embeddings for sample data...")
        resume_emb = get_text_embeddings([sample_resume_text_p])
        job_texts_p = [job['cleaned_text'] for job in sample_jobs_data_p]
        job_embs = get_text_embeddings(job_texts_p)

        if resume_emb is not None and job_embs is not None and resume_emb.shape[0] > 0:
            resume_embedding_single = resume_emb[0]

            print("\n--- Testing Keyword Overlap ---")
            overlap = calculate_keyword_overlap(sample_resume_text_p, sample_jobs_data_p[0]['cleaned_text'],
                                                sample_resume_skills, sample_jobs_data_p[0]['skills'])
            print(f"Keyword overlap with Job 1: {overlap:.4f}")

            print("\n--- Testing Cosine Similarity ---")
            cosine_sim = calculate_cosine_similarity_score(resume_embedding_single, job_embs[0])
            print(f"Cosine similarity with Job 1: {cosine_sim:.4f}")

            print("\n--- Testing Combined Match Score (with SVM placeholder) ---")
            c_score, ind_scores = get_combined_match_score(sample_resume_text_p, sample_jobs_data_p[0]['cleaned_text'],
                                                           resume_embedding_single, job_embs[0],
                                                           sample_resume_skills, sample_jobs_data_p[0]['skills'])
            print(f"Combined score with Job 1: {c_score:.4f}")
            print(f"Individual scores for Job 1: {ind_scores}")

            print("\n--- Matching one resume to multiple jobs ---")
            all_matches = match_resume_to_jobs(sample_resume_text_p, resume_embedding_single, 
                                               sample_jobs_data_p, job_embs,
                                               resume_skills=sample_resume_skills)
            
            print("Top Matches:")
            for job_id, score, individual_details in all_matches[:3]: # Show top 3
                print(f"  Job ID: {job_id}, Combined Score: {score:.4f}")
                # print(f"    Details: {individual_details}")
        else:
            print("Could not generate embeddings for sample data. Skipping matching tests.")
    else:
        print("\nSkipping matching examples as the embedding model could not be loaded.")

    print("\nResume-Job Matching Module execution finished.")
    # Note: The SVM/ANN part is a placeholder. 
    # A real implementation would require: 
    # 1. Labeled data (resume-job pairs زيت (match/no-match)).
    # 2. Feature engineering (e.g., concatenating embeddings, difference, element-wise product, or more complex features).
    # 3. Training the SVM or ANN model.
    # 4. Saving and loading the trained model for prediction.