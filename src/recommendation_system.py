# src/recommendation_system.py
import pandas as pd
import numpy as np

# Assuming other modules are accessible
try:
    from .resume_job_matching import match_resume_to_jobs # For content-based filtering
    from .embedding import get_text_embeddings, EMBEDDING_MODEL, load_embedding_model # For getting embeddings
    from .preprocessing import preprocess_document_text # For processing resume text
except ImportError:
    print("Attempting to import from parent directory for recommendation_system.py")
    from resume_job_matching import match_resume_to_jobs
    from embedding import get_text_embeddings, EMBEDDING_MODEL, load_embedding_model
    from preprocessing import preprocess_document_text

# --- Configuration ---
DEFAULT_N_RECOMMENDATIONS = 5

# --- 1. Content-Based Recommendation (using Match Scores) ---
def recommend_jobs_content_based(candidate_resume_text, all_jobs_data, all_job_embeddings,
                                 n_recommendations=DEFAULT_N_RECOMMENDATIONS):
    """
    Recommends jobs based on content similarity between candidate's resume and job descriptions.

    Args:
        candidate_resume_text (str): Raw text of the candidate's resume.
        all_jobs_data (list of dict): List of job data dictionaries, each should have at least 
                                      'job_id', 'cleaned_text' (or raw text for processing), and 'skills'.
        all_job_embeddings (np.ndarray): A 2D numpy array of embeddings for all job descriptions.
        n_recommendations (int): Number of top jobs to recommend.

    Returns:
        list: A list of recommended job_ids, sorted by relevance.
              Returns empty list if error occurs.
    """
    if EMBEDDING_MODEL is None:
        print("Embedding model not loaded. Attempting to load for recommendations...")
        load_embedding_model()
        if EMBEDDING_MODEL is None:
            print("Failed to load embedding model. Cannot provide content-based recommendations.")
            return []

    if not candidate_resume_text or not all_jobs_data or all_job_embeddings is None:
        print("Error: Missing resume text, job data, or job embeddings for content-based recommendation.")
        return []

    # Preprocess the candidate's resume
    # print("Preprocessing candidate resume for recommendation...")
    processed_resume = preprocess_document_text(candidate_resume_text, document_type='resume')
    if not processed_resume or not processed_resume.get('cleaned_text'):
        print("Error: Could not preprocess resume text.")
        return []
    
    resume_cleaned_text = processed_resume['cleaned_text']
    resume_skills = processed_resume.get('skills', [])

    # Get embedding for the resume
    # print("Generating embedding for candidate resume...")
    resume_embedding_array = get_text_embeddings([resume_cleaned_text])
    if resume_embedding_array is None or resume_embedding_array.shape[0] == 0:
        print("Error: Could not generate embedding for the resume.")
        return []
    resume_embedding = resume_embedding_array[0]

    # Use the match_resume_to_jobs function
    # Ensure all_jobs_data has 'cleaned_text' and 'skills' for matching function
    # If not, they need to be processed here or ensured upstream.
    # For this example, assume all_jobs_data contains preprocessed 'cleaned_text' and 'skills'.
    
    # We need to ensure all_jobs_data has the fields expected by match_resume_to_jobs
    # Specifically: 'cleaned_text', 'skills', 'job_id'
    # Let's assume `all_jobs_data` is a list of dicts like:
    # [{'job_id': 'j1', 'cleaned_text': '...', 'skills': ['python', ...]}, ...]

    print(f"Matching resume to {len(all_jobs_data)} jobs...")
    job_matches = match_resume_to_jobs(
        resume_cleaned_text,
        resume_embedding,
        all_jobs_data, # This should be the list of job dicts
        all_job_embeddings, # This is the matrix of job embeddings
        resume_skills=resume_skills
    )

    if not job_matches:
        print("No matches found or error in matching process.")
        return []

    # Extract top N recommendations (job_id, score, details_dict)
    recommended_jobs_with_scores = job_matches[:n_recommendations]
    
    # Log the recommendations with scores for clarity
    print(f"\nTop {len(recommended_jobs_with_scores)} Content-Based Recommendations:")
    for job_id, score, _ in recommended_jobs_with_scores:
        print(f"  Job ID: {job_id}, Match Score: {score:.4f}")
        
    recommended_job_ids = [match[0] for match in recommended_jobs_with_scores]
    return recommended_job_ids

# --- 2. Collaborative Filtering (Conceptual Placeholder) ---
# This requires user-item interaction data (e.g., user applied to job, user viewed job).
# Common techniques: User-based CF, Item-based CF, Matrix Factorization (e.g., SVD).

def recommend_jobs_collaborative_filtering(user_id, user_item_interaction_matrix, n_recommendations=DEFAULT_N_RECOMMENDATIONS):
    """
    Placeholder for collaborative filtering recommendations.

    Args:
        user_id: The ID of the user for whom to generate recommendations.
        user_item_interaction_matrix (pd.DataFrame or np.ndarray): 
            Matrix where rows are users, columns are jobs, values are interactions (e.g., 1 if applied, 0 otherwise).
        n_recommendations (int): Number of jobs to recommend.

    Returns:
        list: A list of recommended job_ids.
    """
    print("\n--- Collaborative Filtering Recommendation (Conceptual Placeholder) ---")
    print(f"Generating CF recommendations for user {user_id}...")
    # 1. Find users similar to user_id based on interaction patterns.
    # 2. Identify jobs that similar users liked/applied to but user_id has not interacted with.
    # 3. Rank these jobs and return the top N.
    # This is a complex implementation, so we'll just return a placeholder.
    print("Collaborative filtering is not implemented in this basic version.")
    print("It would require user-job interaction data and a CF algorithm (e.g., SVD, KNN).")
    
    # Placeholder: return some random job IDs from the matrix columns if available
    if isinstance(user_item_interaction_matrix, pd.DataFrame) and not user_item_interaction_matrix.empty:
        available_jobs = user_item_interaction_matrix.columns.tolist()
        if len(available_jobs) > n_recommendations:
            return np.random.choice(available_jobs, n_recommendations, replace=False).tolist()
        else:
            return available_jobs
    return [f"cf_placeholder_job_{i+1}" for i in range(n_recommendations)]

# --- 3. Hybrid Recommendation (Combining Content-Based and Collaborative) ---
def recommend_jobs_hybrid(candidate_resume_text, user_id, all_jobs_data, all_job_embeddings, 
                          user_item_interaction_matrix=None, 
                          n_recommendations=DEFAULT_N_RECOMMENDATIONS, 
                          cb_weight=0.7, cf_weight=0.3):
    """
    Combines content-based and collaborative filtering recommendations (conceptual).
    A simple approach is to take recommendations from both and re-rank or merge.

    Args:
        candidate_resume_text (str): Raw resume text.
        user_id: User ID for CF.
        all_jobs_data (list of dict): Data for all jobs.
        all_job_embeddings (np.ndarray): Embeddings for all jobs.
        user_item_interaction_matrix (pd.DataFrame, optional): Interaction data for CF.
        n_recommendations (int): Total number of recommendations.
        cb_weight (float): Weight for content-based scores.
        cf_weight (float): Weight for collaborative filtering scores.

    Returns:
        list: A list of final recommended job_ids.
    """
    print("\n--- Hybrid Recommendation (Conceptual) ---")
    
    # Get content-based recommendations
    cb_recs_ids = recommend_jobs_content_based(candidate_resume_text, all_jobs_data, all_job_embeddings, 
                                               n_recommendations=n_recommendations * 2) # Get more to allow for merging

    cf_recs_ids = []
    if user_item_interaction_matrix is not None and cf_weight > 0:
        cf_recs_ids = recommend_jobs_collaborative_filtering(user_id, user_item_interaction_matrix, 
                                                             n_recommendations=n_recommendations * 2)
    
    # Simple merging strategy: Give priority to CB, fill with CF if needed.
    # A more sophisticated approach would involve re-scoring based on weights.
    final_recommendations = []
    seen_jobs = set()

    for job_id in cb_recs_ids:
        if job_id not in seen_jobs and len(final_recommendations) < n_recommendations:
            final_recommendations.append(job_id)
            seen_jobs.add(job_id)
    
    if cf_recs_ids:
        for job_id in cf_recs_ids:
            if job_id not in seen_jobs and len(final_recommendations) < n_recommendations:
                final_recommendations.append(job_id)
                seen_jobs.add(job_id)
    
    print(f"Final Hybrid Recommendations ({len(final_recommendations)} jobs): {final_recommendations}")
    return final_recommendations[:n_recommendations]

# --- Sample Usage ---
if __name__ == "__main__":
    print("Starting Recommendation System Module...")

    # Sample data (ensure embedding model is loaded if not already)
    if EMBEDDING_MODEL is None:
        print("Recommendation system: Attempting to load embedding model...")
        load_embedding_model()

    if EMBEDDING_MODEL:
        sample_candidate_resume = """
        Highly skilled Python developer with 5 years of experience in web development, 
        data analysis, and machine learning. Proficient in Django, Flask, Pandas, Scikit-learn. 
        Strong background in AWS cloud services and building scalable applications. 
        Seeking a challenging role in a dynamic tech company.
        Skills: Python, Django, Flask, Pandas, Scikit-learn, AWS, SQL, Git.
        """
        
        # Assume these are preprocessed and embeddings generated elsewhere for all_jobs_data
        # For this example, we'll create dummy processed data and embeddings here.
        job_texts_for_embedding = [
            "senior python developer aws cloud microservices backend focus", # Job 1 (Good match)
            "java engineer spring boot sql database design enterprise applications", # Job 2 (Less match)
            "machine learning specialist nlp tensorflow pytorch research python data science", # Job 3 (Good match for ML part)
            "frontend developer react javascript html css responsive web design expert", # Job 4 (Less match)
            "python data engineer etl pipelines airflow big data technologies aws" # Job 5 (Good match for Python/AWS/Data)
        ]
        
        sample_all_jobs_data = [
            {'job_id': 'job1', 'cleaned_text': job_texts_for_embedding[0], 'skills': ['python', 'aws', 'cloud', 'microservices', 'backend']},
            {'job_id': 'job2', 'cleaned_text': job_texts_for_embedding[1], 'skills': ['java', 'spring boot', 'sql', 'database', 'enterprise']},
            {'job_id': 'job3', 'cleaned_text': job_texts_for_embedding[2], 'skills': ['machine learning', 'nlp', 'tensorflow', 'pytorch', 'python', 'research']},
            {'job_id': 'job4', 'cleaned_text': job_texts_for_embedding[3], 'skills': ['react', 'javascript', 'html', 'css', 'frontend']},
            {'job_id': 'job5', 'cleaned_text': job_texts_for_embedding[4], 'skills': ['python', 'etl', 'airflow', 'big data', 'aws']}
        ]
        
        print("\nGenerating embeddings for sample jobs for recommendation module...")
        sample_all_job_embeddings = get_text_embeddings(job_texts_for_embedding)

        if sample_all_job_embeddings is not None and sample_all_job_embeddings.shape[0] > 0:
            print("\n--- Testing Content-Based Recommendations ---")
            cb_recs = recommend_jobs_content_based(sample_candidate_resume, 
                                                   sample_all_jobs_data, 
                                                   sample_all_job_embeddings, 
                                                   n_recommendations=3)
            # print(f"Content-Based Recommended Job IDs: {cb_recs}")

            # --- For Collaborative Filtering and Hybrid (Conceptual) ---
            sample_user_id = "user123"
            # Dummy interaction matrix (users x jobs)
            interaction_data = {
                'job1': [1, 0, 1, 0], # User1 applied, User3 applied
                'job2': [0, 1, 0, 0], # User2 applied
                'job3': [1, 1, 0, 1], # User1, User2, User4 applied
                'job4': [0, 0, 1, 0], # User3 applied
                'job5': [1, 0, 0, 1]  # User1, User4 applied
            }
            users = [f'user{i+1}' for i in range(4)]
            sample_interaction_df = pd.DataFrame(interaction_data, index=users)
            
            # Note: For user_id "user123" (not in matrix), CF would typically use cold-start strategies or fail.
            # For this placeholder, it might return random jobs.
            cf_recs = recommend_jobs_collaborative_filtering(sample_user_id, sample_interaction_df, n_recommendations=3)
            # print(f"Collaborative Filtering Recommended Job IDs (Placeholder): {cf_recs}")

            hybrid_recs = recommend_jobs_hybrid(sample_candidate_resume, sample_user_id, 
                                                sample_all_jobs_data, sample_all_job_embeddings, 
                                                user_item_interaction_matrix=sample_interaction_df, 
                                                n_recommendations=3)
            # print(f"Hybrid Recommended Job IDs (Conceptual): {hybrid_recs}")
        else:
            print("Could not generate job embeddings. Skipping recommendation tests.")
    else:
        print("\nSkipping recommendation examples as the embedding model could not be loaded.")

    print("\nRecommendation System Module execution finished.")
    # Notes:
    # - A real collaborative filtering system needs significant user interaction data.
    # - Hybrid models can be complex, involving weighted scores, switching strategies, or feature augmentation.
    # - Cold-start problem (new users or new jobs) is a key challenge in recommendation systems.
    # - Evaluation of recommendation systems involves metrics like precision@k, recall@k, nDCG, MAP.