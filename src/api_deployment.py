# src/api_deployment.py
import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np

# --- Mocking imports from other project modules ---
# In a real scenario, these would be actual imports and functions.
# Make sure these functions are robust and handle potential errors.

# Placeholder: Assume embedding model and other necessary models are loaded on startup
# This is crucial for performance in a real API.

# from .data_ingestion import ingest_data # For processing uploaded files
# from .preprocessing import preprocess_document_text
# from .embedding import get_text_embeddings, load_embedding_model, EMBEDDING_MODEL
# from .resume_job_matching import match_resume_to_jobs # Using the refined version
# from .salary_forecasting import predict_salary # Assuming a predict_salary function exists
# from .job_classification import predict_job_role # Assuming a predict_job_role function exists
# from .recommendation_system import recommend_jobs_content_based

# --- Mock Implementations (Replace with actual module calls) ---
# These are simplified versions for the API to function without full backend integration yet.

# Load embedding model (conceptual - should happen once at startup)
EMBEDDING_MODEL_API = None
ALL_JOB_DATA_API = [] # List of job dicts: {'job_id', 'title', 'description', 'cleaned_text', 'skills', ...}
ALL_JOB_EMBEDDINGS_API = None # Numpy array of job embeddings
JOB_CLASSIFIER_API = None
SALARY_MODEL_API = None

def load_models_and_data():
    global EMBEDDING_MODEL_API, ALL_JOB_DATA_API, ALL_JOB_EMBEDDINGS_API, JOB_CLASSIFIER_API, SALARY_MODEL_API
    print("API: Attempting to load models and data...")
    # Mock loading embedding model
    # In reality: from .embedding import load_embedding_model; load_embedding_model(); EMBEDDING_MODEL_API = EMBEDDING_MODEL
    EMBEDDING_MODEL_API = "mock_embedding_model_loaded"
    print(f"API: Embedding model status: {EMBEDDING_MODEL_API}")

    # Mock loading job data and embeddings
    # In reality, load from a database or precomputed files
    job_texts_for_embedding = [
        "senior python developer aws cloud microservices backend focus",
        "java engineer spring boot sql database design enterprise applications",
        "machine learning specialist nlp tensorflow pytorch research python data science",
        "frontend developer react javascript html css responsive web design expert",
        "python data engineer etl pipelines airflow big data technologies aws"
    ]
    ALL_JOB_DATA_API = [
        {'job_id': 'job1', 'title': 'Senior Python Developer', 'description': job_texts_for_embedding[0], 'cleaned_text': job_texts_for_embedding[0], 'skills': ['python', 'aws', 'backend'], 'location': 'Remote', 'experience': '5 years'},
        {'job_id': 'job2', 'title': 'Java Engineer', 'description': job_texts_for_embedding[1], 'cleaned_text': job_texts_for_embedding[1], 'skills': ['java', 'spring', 'sql'], 'location': 'New York', 'experience': '3 years'},
        {'job_id': 'job3', 'title': 'Machine Learning Specialist', 'description': job_texts_for_embedding[2], 'cleaned_text': job_texts_for_embedding[2], 'skills': ['ml', 'python', 'nlp'], 'location': 'San Francisco', 'experience': '4 years'},
        {'job_id': 'job4', 'title': 'Frontend Developer', 'description': job_texts_for_embedding[3], 'cleaned_text': job_texts_for_embedding[3], 'skills': ['react', 'javascript', 'css'], 'location': 'Remote', 'experience': '2 years'},
        {'job_id': 'job5', 'title': 'Python Data Engineer', 'description': job_texts_for_embedding[4], 'cleaned_text': job_texts_for_embedding[4], 'skills': ['python', 'etl', 'aws'], 'location': 'Austin', 'experience': '4 years'}
    ]
    # Mock embeddings (replace with actual get_text_embeddings call)
    if EMBEDDING_MODEL_API:
        # ALL_JOB_EMBEDDINGS_API = get_text_embeddings([job['cleaned_text'] for job in ALL_JOB_DATA_API])
        ALL_JOB_EMBEDDINGS_API = np.random.rand(len(ALL_JOB_DATA_API), 384) # Assuming 384 dim for MiniLM
        print(f"API: Mock job embeddings generated, shape: {ALL_JOB_EMBEDDINGS_API.shape}")
    else:
        print("API: Embedding model not loaded, cannot generate job embeddings.")

    # Mock loading job classifier (e.g., from joblib file)
    # In reality: from joblib import load; JOB_CLASSIFIER_API = load('path/to/job_classifier.joblib')
    JOB_CLASSIFIER_API = "mock_job_classifier_loaded"
    print(f"API: Job classifier status: {JOB_CLASSIFIER_API}")

    # Mock loading salary model
    # In reality: from joblib import load; SALARY_MODEL_API = load('path/to/salary_model.joblib')
    SALARY_MODEL_API = "mock_salary_model_loaded"
    print(f"API: Salary model status: {SALARY_MODEL_API}")

# --- FastAPI App Initialization ---
app = FastAPI(title="Resume-Job Matching and Market Insights API")

# Serve static files (for HTML, CSS, JS frontend)
# Create a 'static' directory in the same location as this script if it doesn't exist
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static_frontend")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Temporary storage for uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Pydantic Models for Request/Response Bodies ---
class Job(BaseModel):
    job_id: str
    title: str
    description: str
    skills: list[str]
    match_score: float | None = None

class SalaryPredictionRequest(BaseModel):
    job_title: str | None = None
    skills: list[str]
    experience_years: int
    location: str | None = None
    # Add other relevant features your salary model uses

class SalaryPredictionResponse(BaseModel):
    predicted_salary_min: float
    predicted_salary_max: float
    currency: str = "USD"
    notes: str | None = None

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    load_models_and_data() # Load all necessary models and data when API starts

@app.get("/", response_class=HTMLResponse)
async def get_simple_frontend():
    # Simple HTML form for interaction
    # This HTML file should be placed in the STATIC_DIR
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Pipeline Interface</title>
        <link rel="stylesheet" href="/static/styles.css">
    </head>
    <body>
        <h1>Resume-Job Matching & Insights</h1>

        <h2>1. Upload Resume for Job Recommendations</h2>
        <form id="resumeUploadForm" enctype="multipart/form-data">
            <input type="file" name="resume_file" accept=".pdf,.docx,.txt" required>
            <button type="submit">Get Recommendations</button>
        </form>
        <div id="recommendationsResult"></div>

        <h2>2. Predict Salary</h2>
        <form id="salaryPredictionForm">
            <label for="skills">Skills (comma-separated):</label>
            <input type="text" id="skills" name="skills" value="python,aws,api" required><br><br>
            <label for="experience">Experience (years):</label>
            <input type="number" id="experience" name="experience" value="5" required><br><br>
            <label for="location">Location (optional):</label>
            <input type="text" id="location" name="location" value="Remote"><br><br>
            <button type="submit">Predict Salary</button>
        </form>
        <div id="salaryResult"></div>
        
        <h2>3. Get All Available Jobs (Mock)</h2>
        <button onclick="fetchAllJobs()">Fetch All Jobs</button>
        <div id="allJobsResult"></div>

        <script src="/static/scripts.js"></script>
    </body>
    </html>
    """
    # Create a dummy styles.css and scripts.js in STATIC_DIR for this to work fully
    # For now, embedding basic JS for form handling directly in HTML for simplicity if scripts.js is not made
    # Or, better, create those files.

    # Create static/styles.css
    styles_css_path = os.path.join(STATIC_DIR, "styles.css")
    if not os.path.exists(styles_css_path):
        with open(styles_css_path, "w") as f:
            f.write("""
            body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }
            h1, h2 { color: #333; }
            form { background-color: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            input[type='file'], input[type='text'], input[type='number'] { margin-bottom: 10px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: calc(100% - 20px); }
            button { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            #recommendationsResult, #salaryResult, #allJobsResult { margin-top: 15px; padding: 10px; background-color: #e9e9e9; border-radius: 4px; }
            .job-item { border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 5px; }
            """)

    # Create static/scripts.js
    scripts_js_path = os.path.join(STATIC_DIR, "scripts.js")
    if not os.path.exists(scripts_js_path):
        with open(scripts_js_path, "w") as f:
            f.write("""
            document.getElementById('resumeUploadForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const formData = new FormData(this);
                const resultDiv = document.getElementById('recommendationsResult');
                resultDiv.innerHTML = 'Processing...';
                try {
                    const response = await fetch('/recommend-jobs/', {
                        method: 'POST',
                        body: formData
                    });
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    const data = await response.json();
                    let html = '<h3>Recommended Jobs:</h3>';
                    if (data.recommendations && data.recommendations.length > 0) {
                        data.recommendations.forEach(job => {
                            html += `<div class="job-item"><strong>${job.title} (ID: ${job.job_id})</strong><br>Skills: ${job.skills.join(', ')}<br>Score: ${job.match_score ? job.match_score.toFixed(4) : 'N/A'}</div>`;
                        });
                    } else {
                        html += '<p>No recommendations found or error processing.</p>';
                    }
                    resultDiv.innerHTML = html;
                } catch (error) {
                    resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                }
            });

            document.getElementById('salaryPredictionForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const skills = document.getElementById('skills').value.split(',').map(s => s.trim());
                const experience = parseInt(document.getElementById('experience').value);
                const location = document.getElementById('location').value;
                const resultDiv = document.getElementById('salaryResult');
                resultDiv.innerHTML = 'Predicting...';
                try {
                    const response = await fetch('/predict-salary/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ skills: skills, experience_years: experience, location: location })
                    });
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    const data = await response.json();
                    resultDiv.innerHTML = `<p>Predicted Salary: ${data.predicted_salary_min} - ${data.predicted_salary_max} ${data.currency}</p>${data.notes ? `<p>Notes: ${data.notes}</p>` : ''}`;
                } catch (error) {
                    resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                }
            });
            
            async function fetchAllJobs() {
                const resultDiv = document.getElementById('allJobsResult');
                resultDiv.innerHTML = 'Fetching...';
                try {
                    const response = await fetch('/jobs/');
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    const data = await response.json();
                    let html = '<h3>Available Jobs:</h3>';
                    if (data && data.length > 0) {
                        data.forEach(job => {
                            html += `<div class="job-item"><strong>${job.title} (ID: ${job.job_id})</strong><br>Description: ${job.description.substring(0,100)}...<br>Skills: ${job.skills.join(', ')}</div>`;
                        });
                    } else {
                        html += '<p>No jobs available.</p>';
                    }
                    resultDiv.innerHTML = html;
                } catch (error) {
                    resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                }
            }
            """)
    return HTMLResponse(content=html_content)

@app.post("/upload-resume/")
async def upload_resume(resume_file: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{resume_file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(resume_file.file, buffer)

    # Mock processing: In reality, call data_ingestion and preprocessing
    # raw_text = ingest_data(file_path) # This would need to handle single file path
    # processed_resume = preprocess_document_text(raw_text, document_type='resume')
    # For now, just simulate
    try:
        with open(file_path, 'r', encoding='utf-8') as f: # Assuming text file for mock
            resume_text_content = f.read(500) # Read first 500 chars for mock
    except Exception:
        resume_text_content = "Mock resume content from non-text file."

    # Clean up uploaded file (optional, or use a scheduled task)
    # os.remove(file_path)

    return {
        "message": "Resume uploaded successfully (mock processing)", 
        "file_id": file_id, 
        "filename": resume_file.filename,
        "mock_content_preview": resume_text_content,
        "cleaned_text_mock": "mock cleaned resume text...",
        "skills_mock": ["python", "fastapi", "mocking"]
    }

@app.post("/recommend-jobs/", response_model=dict[str, list[Job] | str])
async def recommend_jobs_endpoint(resume_file: UploadFile = File(...)):
    if not EMBEDDING_MODEL_API or ALL_JOB_EMBEDDINGS_API is None or not ALL_JOB_DATA_API:
        raise HTTPException(status_code=503, detail="Models or job data not ready. Please try again later.")

    # Save and process resume
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_resume_for_rec.tmp")
    resume_text = ""
    try:
        async with resume_file.open("rb") as buffer: # Use async open for UploadFile
            content_bytes = await buffer.read()
        # Basic text extraction (highly simplified - use data_ingestion for real cases)
        if resume_file.filename.endswith('.txt'):
            resume_text = content_bytes.decode('utf-8', errors='ignore')
        elif resume_file.filename.endswith('.pdf'):
            resume_text = "Mock PDF text: python developer with api experience" # Placeholder
        elif resume_file.filename.endswith('.docx'):
            resume_text = "Mock DOCX text: data scientist skilled in machine learning" # Placeholder
        else:
            resume_text = "Unsupported file type for mock processing. Content: python, java, problem solving."
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from resume or resume is empty.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume file: {str(e)}")
    finally:
        if os.path.exists(file_path): os.remove(file_path) # Clean up

    # Mock preprocessing and embedding for the resume
    # processed_resume = preprocess_document_text(resume_text, 'resume')
    # resume_cleaned_text = processed_resume['cleaned_text']
    # resume_skills = processed_resume['skills']
    # resume_embedding = get_text_embeddings([resume_cleaned_text])[0]
    
    # MOCKING these steps:
    resume_cleaned_text = resume_text.lower() # Simplified
    resume_skills_mock = [s.strip() for s in resume_cleaned_text.split() if len(s)>2][:5] # very basic skill extraction
    # resume_embedding_mock = get_text_embeddings([resume_cleaned_text])[0] if EMBEDDING_MODEL_API else np.random.rand(384)
    resume_embedding_mock = np.random.rand(384) # Mock embedding

    # Mock call to recommendation logic (content-based)
    # recommended_job_ids = recommend_jobs_content_based(resume_cleaned_text, ALL_JOB_DATA_API, ALL_JOB_EMBEDDINGS_API)
    # For mock, let's use cosine similarity directly here for simplicity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(resume_embedding_mock.reshape(1, -1), ALL_JOB_EMBEDDINGS_API)[0]
    
    # Get top N matches
    top_n_indices = np.argsort(similarities)[::-1][:5] # Top 5
    
    recommendations = []
    for i in top_n_indices:
        job = ALL_JOB_DATA_API[i]
        recommendations.append(Job(
            job_id=job['job_id'], 
            title=job['title'], 
            description=job['description'], 
            skills=job['skills'],
            match_score=float(similarities[i])
        ))

    if not recommendations:
        return {"message": "No suitable job recommendations found based on the mock processing.", "recommendations": []}
    
    return {"recommendations": recommendations}

@app.post("/predict-salary/", response_model=SalaryPredictionResponse)
async def predict_salary_endpoint(request: SalaryPredictionRequest):
    if not SALARY_MODEL_API:
        raise HTTPException(status_code=503, detail="Salary prediction model not ready.")
    
    # Mock prediction. In reality, prepare features and call SALARY_MODEL_API.predict()
    # features = preprocess_for_salary_model(request.skills, request.experience_years, request.location)
    # predicted_range = SALARY_MODEL_API.predict(features)
    
    # Simple mock logic based on experience and number of skills
    base_salary = 40000 + (request.experience_years * 10000) + (len(request.skills) * 2000)
    if "aws" in request.skills or "cloud" in request.skills: base_salary += 5000
    if "python" in request.skills: base_salary += 3000
    if request.location and "Francisco" in request.location : base_salary *=1.2

    return SalaryPredictionResponse(
        predicted_salary_min=float(base_salary * 0.9),
        predicted_salary_max=float(base_salary * 1.1),
        notes="Mock prediction based on experience and skills. For demonstration only."
    )

@app.get("/jobs/", response_model=list[Job])
async def get_all_jobs():
    """Returns a list of all available (mock) jobs."""
    if not ALL_JOB_DATA_API:
        return []
    # Convert to Job model, excluding match_score as it's not relevant here
    return [Job(job_id=j['job_id'], title=j['title'], description=j['description'], skills=j['skills']) for j in ALL_JOB_DATA_API]

# --- Main execution for running the API ---
if __name__ == "__main__":
    print("Starting FastAPI server for ML Pipeline...")
    # To run this: uvicorn api_deployment:app --reload --port 8000
    # The --reload flag is for development, remove for production.
    # The script will try to run it directly if uvicorn is installed.
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("Uvicorn is not installed. To run the API, please install it (`pip install uvicorn`) ")
        print("and then run: `uvicorn src.api_deployment:app --reload --port 8000` from the project root directory.")
    except Exception as e:
        print(f"Could not start Uvicorn server: {e}")
        print("Please ensure Uvicorn is installed and run manually if needed:")
        print("`uvicorn src.api_deployment:app --reload --port 8000` (from project root)")

    # Cleanup temp_uploads directory on exit (optional, can be done manually or via cron)
    # This is a bit tricky to do reliably on app shutdown, especially with --reload.
    # For now, manual cleanup or a separate script is safer for temp_uploads.