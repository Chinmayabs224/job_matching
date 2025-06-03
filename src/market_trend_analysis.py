# src/market_trend_analysis.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Assuming embedding.py might be used for job embeddings
try:
    from .embedding import get_text_embeddings, EMBEDDING_MODEL, load_embedding_model
except ImportError:
    print("Attempting to import from parent directory for market_trend_analysis.py")
    from embedding import get_text_embeddings, EMBEDDING_MODEL, load_embedding_model

# --- Configuration ---
KMEANS_MODEL_SAVE_PATH = "models/market_trends_kmeans.joblib"
N_CLUSTERS_DEFAULT = 5 # Default number of market segments to identify

# --- 1. Job Clustering (using Embeddings or TF-IDF) ---
def cluster_jobs_kmeans(job_features, n_clusters=N_CLUSTERS_DEFAULT, feature_type='embeddings'):
    """
    Clusters job descriptions using KMeans.

    Args:
        job_features (np.ndarray or pd.DataFrame): 
            If feature_type is 'embeddings', this should be a 2D numpy array of job embeddings.
            If feature_type is 'tfidf', this should be a list/Series of preprocessed job texts.
        n_clusters (int): The number of clusters (market segments) to find.
        feature_type (str): 'embeddings' or 'tfidf'. Determines how features are handled.

    Returns:
        sklearn.cluster.KMeans: The fitted KMeans model object.
        np.ndarray: Array of cluster labels assigned to each job.
        np.ndarray: The feature matrix used for clustering (embeddings or TF-IDF vectors).
                  Returns None, None, None if an error occurs.
    """
    if job_features is None or len(job_features) == 0:
        print("Error: Job features are empty.")
        return None, None, None
    
    if len(job_features) < n_clusters:
        print(f"Warning: Number of samples ({len(job_features)}) is less than n_clusters ({n_clusters}). Setting n_clusters to {len(job_features)}.")
        n_clusters = len(job_features)
        if n_clusters == 0:
            print("Error: No samples to cluster after adjustment.")
            return None, None, None

    feature_matrix = None
    
    if feature_type == 'embeddings':
        if not isinstance(job_features, np.ndarray) or job_features.ndim != 2:
            print("Error: For 'embeddings' feature_type, job_features must be a 2D numpy array.")
            return None, None, None
        feature_matrix = StandardScaler().fit_transform(job_features) # Standardize embeddings
    elif feature_type == 'tfidf':
        if not (isinstance(job_features, (list, pd.Series)) and all(isinstance(text, str) for text in job_features)):
            print("Error: For 'tfidf' feature_type, job_features must be a list or Series of strings.")
            return None, None, None
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=5, max_df=0.7)
        try:
            feature_matrix = vectorizer.fit_transform(job_features).toarray() # Convert sparse to dense for KMeans
        except ValueError as e:
            print(f"Error during TF-IDF vectorization (possibly too few documents or features): {e}")
            return None, None, None
        if feature_matrix.shape[0] == 0: # Check if vectorization produced empty output
             print("Error: TF-IDF vectorization resulted in an empty feature matrix.")
             return None, None, None
    else:
        print(f"Error: Unsupported feature_type '{feature_type}'. Choose 'embeddings' or 'tfidf'.")
        return None, None, None

    print(f"Clustering {feature_matrix.shape[0]} jobs into {n_clusters} segments using KMeans ({feature_type})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    try:
        cluster_labels = kmeans.fit_predict(feature_matrix)
    except Exception as e:
        print(f"Error during KMeans fitting: {e}")
        return None, None, None

    # Save the KMeans model (optional)
    # import joblib
    # os.makedirs(os.path.dirname(KMEANS_MODEL_SAVE_PATH), exist_ok=True)
    # joblib.dump(kmeans, KMEANS_MODEL_SAVE_PATH)
    # print(f"KMeans model saved to {KMEANS_MODEL_SAVE_PATH}")

    return kmeans, cluster_labels, feature_matrix

def get_cluster_top_terms(job_texts_list, cluster_labels, n_terms=10):
    """
    Identifies top terms for each cluster using TF-IDF on the texts within each cluster.
    Args:
        job_texts_list (list of str): List of original or preprocessed job texts.
        cluster_labels (np.ndarray): Cluster labels for each job text.
        n_terms (int): Number of top terms to extract per cluster.

    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of top terms.
    """
    if len(job_texts_list) != len(cluster_labels):
        print("Error: Mismatch between length of job texts and cluster labels.")
        return {}

    df = pd.DataFrame({'text': job_texts_list, 'cluster': cluster_labels})
    cluster_top_terms = {}
    
    for i in range(max(cluster_labels) + 1):
        cluster_texts = df[df['cluster'] == i]['text'].tolist()
        if not cluster_texts:
            cluster_top_terms[i] = ["(No documents in cluster)"]
            continue
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1,2))
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores for each term across all documents in the cluster
            summed_tfidf = tfidf_matrix.sum(axis=0)
            if isinstance(summed_tfidf, np.matrix):
                 summed_tfidf = summed_tfidf.A1 # Convert matrix to 1D array
            
            # Get indices of top N terms
            top_indices = summed_tfidf.argsort()[-n_terms:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            cluster_top_terms[i] = top_terms
        except ValueError as e: # Happens if cluster_texts is too small or uniform
            print(f"Could not compute top terms for cluster {i} (texts might be too few/uniform): {e}")
            cluster_top_terms[i] = ["(Error computing terms)"]
            
    return cluster_top_terms

# --- 2. Skill Trend Analysis (Conceptual Placeholder) ---
def analyze_skill_trends(historical_job_data_df, date_column='date_posted', skills_column='extracted_skills'):
    """
    Analyzes skill demand trends over time from historical job data.
    This is a conceptual placeholder.

    Args:
        historical_job_data_df (pd.DataFrame): DataFrame with job postings, 
                                               including a date column and a column with lists of skills.
        date_column (str): Name of the column with posting dates (datetime objects).
        skills_column (str): Name of the column containing lists of skills for each job.

    Returns:
        pd.DataFrame: DataFrame summarizing skill trends (e.g., skill counts per month/year).
                      Returns None if an error occurs.
    """
    print("\n--- Skill Trend Analysis (Conceptual) ---")
    if not isinstance(historical_job_data_df, pd.DataFrame) or \
       date_column not in historical_job_data_df.columns or \
       skills_column not in historical_job_data_df.columns:
        print(f"Error: DataFrame must contain '{date_column}' and '{skills_column}'.")
        return None

    if historical_job_data_df.empty:
        print("Error: Input DataFrame for skill trend analysis is empty.")
        return None

    try:
        # Ensure date column is datetime
        df = historical_job_data_df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
    except Exception as e:
        print(f"Error processing date column: {e}")
        return None

    # Explode skills list into separate rows and count occurrences per time period (e.g., monthly)
    all_skills_over_time = []
    for period_start, group in df.groupby(pd.Grouper(freq='M')):
        period_skills = []
        for skills_list in group[skills_column]:
            if isinstance(skills_list, list):
                period_skills.extend([skill.lower() for skill in skills_list])
        skill_counts = Counter(period_skills)
        for skill, count in skill_counts.items():
            all_skills_over_time.append({'period': period_start.strftime('%Y-%m'), 'skill': skill, 'count': count})
    
    if not all_skills_over_time:
        print("No skills found or extracted to analyze trends.")
        return pd.DataFrame()

    skill_trends_df = pd.DataFrame(all_skills_over_time)
    print("Sample Skill Trends (Top 5 for the first period found):")
    if not skill_trends_df.empty:
        first_period = skill_trends_df['period'].min()
        print(skill_trends_df[skill_trends_df['period'] == first_period].sort_values(by='count', ascending=False).head())
    else:
        print("No skill trend data generated.")
        
    # Further analysis could involve plotting trends for specific skills, identifying emerging/declining skills.
    return skill_trends_df

# --- Sample Usage ---
if __name__ == "__main__":
    print("Starting Market Trend Analysis Module...")

    # Sample job data (preprocessed texts or embeddings)
    sample_job_texts = [
        "senior python developer with aws and microservices experience needed for backend systems",
        "java engineer spring boot sql database design for enterprise applications",
        "machine learning specialist nlp tensorflow pytorch research and development focus",
        "data scientist python r statistics modeling and data visualization required",
        "frontend developer react javascript html css responsive web design expert",
        "full stack engineer nodejs python angular cloud deployment skills important",
        "devops engineer kubernetes docker ci cd pipeline automation and infrastructure management",
        "ui ux designer figma sketch user research and interactive prototype development",
        "product manager agile methodology roadmap planning user stories and market analysis",
        "software architect system design scalable solutions leadership and technical guidance"
    ] * 5 # Multiply to get more data points for clustering

    # Option 1: Use pre-computed embeddings (if available and EMBEDDING_MODEL is loaded)
    job_embeddings = None
    if EMBEDDING_MODEL is None:
        print("\nAttempting to load embedding model for market trend analysis...")
        load_embedding_model()
    
    if EMBEDDING_MODEL:
        print("\nGenerating embeddings for sample job texts...")
        job_embeddings = get_text_embeddings(sample_job_texts)
    
    if job_embeddings is not None and job_embeddings.shape[0] > 0:
        print("\n--- Clustering Jobs using Embeddings ---")
        num_segments = 3 # Example number of segments
        kmeans_model_emb, labels_emb, features_emb = cluster_jobs_kmeans(job_embeddings, n_clusters=num_segments, feature_type='embeddings')
        if kmeans_model_emb:
            print(f"Job cluster labels (Embeddings): {labels_emb[:15]}...")
            # Get top terms for each cluster (using original texts)
            cluster_terms_emb = get_cluster_top_terms(sample_job_texts, labels_emb, n_terms=5)
            print("\nTop terms per cluster (from Embeddings):")
            for cluster_id, terms in cluster_terms_emb.items():
                print(f"  Cluster {cluster_id}: {', '.join(terms)}")
    else:
        print("\nSkipping clustering with embeddings as they could not be generated.")

    # Option 2: Use TF-IDF features directly from texts
    print("\n--- Clustering Jobs using TF-IDF ---")
    num_segments_tfidf = 3
    # Ensure enough samples for TF-IDF min_df
    if len(sample_job_texts) >= num_segments_tfidf * 5: # Heuristic: at least 5 docs per potential cluster for min_df=5
        kmeans_model_tfidf, labels_tfidf, features_tfidf = cluster_jobs_kmeans(sample_job_texts, n_clusters=num_segments_tfidf, feature_type='tfidf')
        if kmeans_model_tfidf:
            print(f"Job cluster labels (TF-IDF): {labels_tfidf[:15]}...")
            cluster_terms_tfidf = get_cluster_top_terms(sample_job_texts, labels_tfidf, n_terms=5)
            print("\nTop terms per cluster (from TF-IDF):")
            for cluster_id, terms in cluster_terms_tfidf.items():
                print(f"  Cluster {cluster_id}: {', '.join(terms)}")
    else:
        print("Skipping TF-IDF clustering due to insufficient sample data for robust TF-IDF vectorization with min_df.")


    # Sample data for skill trend analysis (conceptual)
    historical_data = {
        'job_id': ['j1', 'j2', 'j3', 'j4', 'j5', 'j6'],
        'date_posted': pd.to_datetime(['2023-01-15', '2023-01-20', '2023-02-10', '2023-02-25', '2023-03-05', '2023-03-15']),
        'extracted_skills': [
            ['python', 'aws', 'docker'], 
            ['java', 'spring', 'sql'], 
            ['python', 'machine learning', 'tensorflow'],
            ['aws', 'kubernetes', 'ci/cd'],
            ['python', 'docker', 'react'],
            ['java', 'kafka', 'microservices']
        ]
    }
    historical_df = pd.DataFrame(historical_data)
    skill_trends_summary = analyze_skill_trends(historical_df.copy())
    # if skill_trends_summary is not None and not skill_trends_summary.empty:
    #     print("\nSkill Trend Summary (sample):")
    #     print(skill_trends_summary.head())

    print("\nMarket Trend Analysis Module execution finished.")
    # Notes:
    # - The number of clusters (k) for KMeans is often chosen using methods like the Elbow method or Silhouette score.
    # - Interpreting clusters requires domain knowledge. Top terms help, but manual review is often needed.
    # - Skill trend analysis needs a good volume of time-stamped job data and robust skill extraction.