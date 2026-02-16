from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob
from fastapi.middleware.cors import CORSMiddleware
import time
import random
# NO nltk downloads here!

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# 2. Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SECURITY & CORS ---
# Allows the Next.js frontend (Port 3000) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEURAL ENGINE INIT ---
print("Loading Neural Engine...")
# Using DistilBERT for fast, lightweight sentiment analysis
analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print("Neural Engine Online.")

# --- HELPER FUNCTIONS ---
def extract_keywords(text: str) -> str:
    """Fallback keyword extraction for SQLite string storage"""
    words = [w.strip() for w in text.split() if len(w) > 3]
    return ",".join(words[:3]) if words else "General"

def perform_analysis(text: str):
    """Core analysis engine mapping BERT outputs to the database schema"""
    # Cap text length to prevent BERT token limit crashes (512 max)
    safe_text = text[:500] 
    result = analyzer(safe_text)[0]
    label = result['label']
    confidence = result['score']
    
    # Map strictly positive/negative to a numeric float score
    score = confidence if label == "POSITIVE" else -confidence
    
    return {
        "text": text,
        "score": score,
        "label": label,
        "confidence": confidence,
        "keywords": extract_keywords(text)
    }

# --- Pydantic Models ---
class ListRequest(BaseModel):
    items: List[str]
    
class ScrapeRequest(BaseModel):
    url: str

# --- ENDPOINT 1: TEXT ENTRY ---
@app.post("/analyze_list")
async def analyze_list(req: ListRequest):
    results = [perform_analysis(item) for item in req.items]
    return {
        "total_scanned": len(results),
        "results": results
    }

# --- ENDPOINT 2: FILE UPLOAD (WITH SAFETY CAP) ---
@app.post("/analyze_bulk")
async def analyze_file(file: UploadFile = File(...)):
    start = time.time()
    
    # Read the file safely into memory
    content = await file.read()
    try:
        decoded_content = content.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return {"error": "Invalid file encoding. Use UTF-8."}

    results = []
    MAX_ROWS = 500 # The Safety Cap to prevent server timeouts
    
    for line in decoded_content:
        # Halt execution if we hit the limit
        if len(results) >= MAX_ROWS:
            break
            
        if line.strip():
            # Basic CSV split fallback: take the first column if commas exist
            text = line.split(",")[0] if "," in line else line
            if len(text.strip()) > 5:
                results.append(perform_analysis(text))

    return {
        "total_scanned": len(results),
        "results": results, 
        "meta": {
            "processing_time": round(time.time() - start, 4),
            "warning": f"Capped at {MAX_ROWS} rows to prevent timeout." if len(decoded_content) > MAX_ROWS else None
        }
    }
    
# --- ENDPOINT 3: WEB SCRAPER (UPGRADED TO BYPASS BLOCKS) ---
@app.post("/scrape")
async def scrape_url(req: ScrapeRequest):
    try:
        # 1. SPOOF THE USER-AGENT
        # This tricks websites into thinking we are a real Chrome browser on Windows
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Pass the headers into the request
        res = requests.get(req.url, headers=headers, timeout=10)
        res.raise_for_status() # This will immediately catch and log 403 Forbidden errors
        
        soup = bs4.BeautifulSoup(res.text, 'html.parser')
        
        # Extract paragraphs that actually contain meaningful content
        paragraphs = [p.text.strip() for p in soup.find_all('p') if len(p.text.strip()) > 20]
        
        # Cap scraping to avoid overwhelming the model on huge sites
        paragraphs = paragraphs[:50]
        
        # Safety check if the page blocked us or had no <p> tags
        if not paragraphs:
            return {"error": "No readable paragraph text found on this URL.", "results": []}
            
        results = [perform_analysis(p) for p in paragraphs]
        
        return {
            "total_scanned": len(results),
            "results": results
        }
    except Exception as e:
        # Return the exact error to the frontend so we aren't guessing
        return {"error": f"Scraping failed: {str(e)}", "results": []}