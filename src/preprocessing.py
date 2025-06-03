# src/preprocessing.py
import spacy
import re
import string

# --- Load spaCy model ---
# Make sure to download the model first: python -m spacy download en_core_web_sm
NLP = None
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    # As a fallback, create a blank English model, which won't do NER but allows tokenization.
    NLP = spacy.blank("en") 

# --- Constants ---
# Common skills keywords (extend this list significantly for better accuracy)
# This is a very basic list and should be expanded or replaced with a more robust skill extraction method.
SKILL_KEYWORDS = [
    'python', 'java', 'c++', 'javascript', 'sql', 'nosql', 'mongodb', 'react', 'angular', 'vue',
    'machine learning', 'deep learning', 'natural language processing', 'nlp', 'data analysis',
    'data science', 'statistics', 'computer vision', 'artificial intelligence', 'ai',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'git', 'agile', 'scrum',
    'project management', 'communication', 'teamwork', 'problem solving', 'leadership',
    'html', 'css', 'django', 'flask', 'spring', 'api', 'restful services', 'microservices',
    'big data', 'hadoop', 'spark', 'data visualization', 'tableau', 'power bi', 'excel'
]

# --- Text Cleaning Functions ---
def clean_text(text):
    """Basic text cleaning: lowercase, remove punctuation, extra whitespace."""
    if not isinstance(text, str):
        # print(f"Warning: clean_text received non-string input: {type(text)}. Returning empty string.")
        return "" # Or handle as appropriate, e.g., convert if possible or raise error
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation.replace('-', ''))) # Keep hyphens for compound words
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    text = re.sub(r'\n+', ' ', text) # Replace newlines with spaces
    text = re.sub(r'\t+', ' ', text) # Replace tabs with spaces
    return text

# --- NER and Information Extraction Functions ---
def extract_entities_spacy(text, entity_labels=['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'MONEY', 'PRODUCT']):
    """Extracts named entities using spaCy."""
    if not NLP or not text or not hasattr(NLP, 'pipe'): # Check if NLP model is loaded and has pipe attribute
        return {label: [] for label in entity_labels}
    
    doc = NLP(text)
    entities = {label: [] for label in entity_labels}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

def extract_skills_keywords(text, skill_keywords=SKILL_KEYWORDS):
    """Extracts skills based on a predefined list of keywords (case-insensitive)."""
    if not isinstance(text, str):
        return []
    
    found_skills = set()
    # Use regex to find whole words to avoid partial matches (e.g., 'java' in 'javascript')
    # The \b ensures word boundaries.
    for skill in skill_keywords:
        # Escape special regex characters in skill keyword
        escaped_skill = re.escape(skill)
        if re.search(r'\b' + escaped_skill + r'\b', text, re.IGNORECASE):
            found_skills.add(skill.lower()) # Store in a consistent case
    return sorted(list(found_skills))

def extract_emails(text):
    """Extracts email addresses from text."""
    if not isinstance(text, str):
        return []
    return re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)

def extract_phone_numbers(text):
    """Extracts phone numbers from text (basic US-like patterns)."""
    if not isinstance(text, str):
        return []
    # This regex is a simplified example and might need to be more robust for international numbers
    # It aims to capture common US phone number formats.
    pattern = r'\b(?:\+?1[ -]?)?(?:\(?([2-9][0-8][0-9])\)?[-.●\s]?)?([2-9][0-9]{2})[-.●\s]?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?\b'
    matches = re.finditer(pattern, text)
    return [match.group(0) for match in matches]

# --- Main Preprocessing Function ---
def preprocess_document_text(raw_text, document_type='resume'): # document_type can be 'resume' or 'job_posting'
    """
    Applies a full preprocessing pipeline to raw text from a document.
    Returns a dictionary with cleaned text and extracted features.
    """
    if not isinstance(raw_text, str):
        # print(f"Warning: preprocess_document_text received non-string input: {type(raw_text)}. Skipping.")
        return {
            "cleaned_text": "",
            "entities": {},
            "skills": [],
            "emails": [],
            "phone_numbers": []
        }

    cleaned_text = clean_text(raw_text)
    
    # NER with spaCy (if model is available)
    entities = {}
    if NLP and NLP.meta.get("name", "") != "blank": # Check if it's not a blank model
        entities = extract_entities_spacy(cleaned_text)
    else:
        # Fallback if spaCy model is not loaded or is blank
        entities = {label: [] for label in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'MONEY', 'PRODUCT']}

    # Skill extraction (can be combined with or enhanced by NER results)
    skills = extract_skills_keywords(cleaned_text) # Use cleaned_text for skill extraction
    
    # Contact info extraction
    emails = extract_emails(raw_text) # Use raw_text for emails to preserve case if needed, or cleaned_text
    phone_numbers = extract_phone_numbers(raw_text) # Use raw_text for phone numbers

    return {
        "original_text_preview": raw_text[:200] + "..." if len(raw_text) > 200 else raw_text,
        "cleaned_text": cleaned_text,
        "entities": entities, # e.g., {'GPE': ['new york', 'london'], 'ORG': ['google']}
        "skills": skills,     # e.g., ['python', 'machine learning']
        "emails": emails,
        "phone_numbers": phone_numbers
    }

# --- Sample Usage ---
if __name__ == "__main__":
    print("Starting Preprocessing Module...")

    sample_resume_text = """
    John Doe
    New York, NY | (123) 456-7890 | john.doe@email.com

    Summary
    Highly skilled and motivated software engineer with 5+ years of experience in Python, Java, and Machine Learning.
    Proven ability to develop scalable applications and work effectively in Agile teams. 
    Worked at Google and Microsoft.
    Seeking a challenging role at a forward-thinking company.
    Expected Salary: $120,000 per year.
    Project X: Developed a NLP model for sentiment analysis.
    """

    sample_job_description_text = """
    Job Title: Senior Python Developer
    Company: Innovatech Ltd.
    Location: San Francisco, CA
    Salary: $130,000 - $160,000

    We are looking for an experienced Python Developer to join our dynamic team. 
    Responsibilities include designing and implementing software solutions. 
    Must have strong experience with Django, Flask, and RESTful APIs. 
    Knowledge of AWS and Docker is a plus. 
    The candidate should have excellent communication skills.
    Contact: hiring@innovatech.com or call 555-123-4567 ext. 101
    """

    print("\n--- Preprocessing Sample Resume ---")
    if NLP:
        processed_resume = preprocess_document_text(sample_resume_text, document_type='resume')
        print(f"Cleaned Text (preview): {processed_resume['cleaned_text'][:100]}...")
        print(f"Extracted Entities (GPE): {processed_resume['entities'].get('GPE', [])}")
        print(f"Extracted Entities (ORG): {processed_resume['entities'].get('ORG', [])}")
        print(f"Extracted Skills: {processed_resume['skills']}")
        print(f"Extracted Emails: {processed_resume['emails']}")
        print(f"Extracted Phone Numbers: {processed_resume['phone_numbers']}")
    else:
        print("Skipping resume preprocessing as spaCy model is not available.")

    print("\n--- Preprocessing Sample Job Description ---")
    if NLP:
        processed_job_desc = preprocess_document_text(sample_job_description_text, document_type='job_posting')
        print(f"Cleaned Text (preview): {processed_job_desc['cleaned_text'][:100]}...")
        print(f"Extracted Entities (LOC): {processed_job_desc['entities'].get('LOC', [])}") # Location
        print(f"Extracted Entities (ORG): {processed_job_desc['entities'].get('ORG', [])}") # Organization
        print(f"Extracted Skills: {processed_job_desc['skills']}")
        print(f"Extracted Emails: {processed_job_desc['emails']}")
        print(f"Extracted Phone Numbers: {processed_job_desc['phone_numbers']}")
    else:
        print("Skipping job description preprocessing as spaCy model is not available.")

    # Example with non-string input
    print("\n--- Preprocessing Non-string Input (should be handled gracefully) ---")
    processed_invalid = preprocess_document_text(12345)
    print(f"Processed Invalid: {processed_invalid}")

    print("\nPreprocessing Module execution finished.")

    # Note: For a real application, the SKILL_KEYWORDS list needs to be much more comprehensive
    # or replaced/augmented with a more sophisticated skill extraction technique (e.g., training a custom NER model for skills).
    # Also, location extraction might need normalization (e.g., 'sf' -> 'San Francisco').