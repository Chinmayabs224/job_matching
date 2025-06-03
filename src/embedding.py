# src/embedding.py
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Configuration ---
# You can choose different pre-trained models from sentence-transformers.
# 'all-MiniLM-L6-v2' is a good starting point: fast and decent quality.
# Other options: 'all-mpnet-base-v2' (higher quality, slower), 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual)
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_MODEL = None

# --- Load Embedding Model ---
def load_embedding_model(model_name=MODEL_NAME):
    """Loads the Sentence-BERT model."""
    global EMBEDDING_MODEL
    try:
        EMBEDDING_MODEL = SentenceTransformer(model_name)
        print(f"Successfully loaded SentenceTransformer model: {model_name}")
    except Exception as e:
        print(f"Error loading SentenceTransformer model '{model_name}': {e}")
        print("Please ensure 'sentence-transformers' is installed and the model name is correct.")
        EMBEDDING_MODEL = None
    return EMBEDDING_MODEL

# --- Embedding Generation Function ---
def get_text_embeddings(texts_list, model=None):
    """
    Generates embeddings for a list of texts using the pre-loaded Sentence-BERT model.
    
    Args:
        texts_list (list of str): A list of text strings to embed.
        model (SentenceTransformer, optional): The loaded SentenceTransformer model. 
                                              If None, uses the globally loaded EMBEDDING_MODEL.

    Returns:
        numpy.ndarray: A 2D numpy array where each row is the embedding for the corresponding text.
                       Returns None if the model is not loaded or an error occurs.
    """
    active_model = model if model else EMBEDDING_MODEL
    
    if active_model is None:
        print("Embedding model is not loaded. Cannot generate embeddings.")
        # Attempt to load the default model if it hasn't been loaded yet
        print("Attempting to load the default embedding model...")
        if load_embedding_model(): # Try loading the default model
            active_model = EMBEDDING_MODEL
        else:
            return None
            
    if not isinstance(texts_list, list) or not all(isinstance(text, str) for text in texts_list):
        print("Input must be a list of strings.")
        return None
    
    if not texts_list:
        print("Input text list is empty.")
        return np.array([]) # Return an empty array for empty input

    try:
        print(f"Generating embeddings for {len(texts_list)} texts...")
        embeddings = active_model.encode(texts_list, convert_to_numpy=True, show_progress_bar=True)
        print(f"Successfully generated embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

# --- Sample Usage ---
if __name__ == "__main__":
    print("Starting Embedding Module...")

    # Ensure the model is loaded (it's also called in get_text_embeddings if not loaded)
    if EMBEDDING_MODEL is None:
        load_embedding_model()

    if EMBEDDING_MODEL:
        sample_resumes_text = [
            "Experienced software engineer with strong Python and Java skills. Worked on machine learning projects.",
            "Recent graduate with a degree in computer science, eager to learn and contribute. Knows C++ and SQL.",
            "Data scientist specializing in natural language processing and deep learning. Proficient in TensorFlow and PyTorch."
        ]

        sample_job_descriptions_text = [
            "Seeking a senior software developer proficient in Python and cloud technologies like AWS.",
            "Entry-level software developer position. Requires knowledge of Java or C++ and good problem-solving skills.",
            "Join our AI team as an NLP engineer. Must have experience with BERT and transformer models."
        ]

        print("\n--- Generating Embeddings for Sample Resumes ---")
        resume_embeddings = get_text_embeddings(sample_resumes_text)
        if resume_embeddings is not None:
            print(f"Shape of resume embeddings: {resume_embeddings.shape}")
            # print(f"First resume embedding (first 5 dimensions): {resume_embeddings[0][:5]}...")

        print("\n--- Generating Embeddings for Sample Job Descriptions ---")
        job_embeddings = get_text_embeddings(sample_job_descriptions_text)
        if job_embeddings is not None:
            print(f"Shape of job embeddings: {job_embeddings.shape}")
            # print(f"First job description embedding (first 5 dimensions): {job_embeddings[0][:5]}...")
        
        # Example of handling empty list
        print("\n--- Generating Embeddings for Empty List ---")
        empty_embeddings = get_text_embeddings([])
        if empty_embeddings is not None:
            print(f"Shape of empty embeddings: {empty_embeddings.shape}")

        # Example of handling non-list input (should print an error)
        print("\n--- Generating Embeddings for Invalid Input ---")
        invalid_embeddings = get_text_embeddings("This is not a list")
        if invalid_embeddings is None:
            print("Correctly handled invalid input type.")

    else:
        print("\nSkipping embedding generation examples as the model could not be loaded.")

    print("\nEmbedding Module execution finished.")

    # Note: The actual embedding vectors are high-dimensional (e.g., 384 for 'all-MiniLM-L6-v2').
    # These embeddings can then be used for similarity calculations, clustering, etc.