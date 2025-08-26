# app.py (Updated with new prompt)
import os
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import PIL.Image
import io

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

MODEL_NAME = "gemini-2.5-flash-lite"

# --- NEW, HYPER-SPECIFIC SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a world-class car specialist. Your task is to identify as many cars as possible in the provided image.

For each car you identify, you MUST follow this exact Markdown format:

### **[Unique Visual Identifier]**
- **Identification:** [Make Model (Estimated Year Range)]
- **Reasoning:** [Briefly explain the key visual cues like headlight shape, grille, body lines, or unique features that led to your identification.]

A "Unique Visual Identifier" MUST be a clear, simple description of the car's location or appearance in the image (e.g., "The red SUV in the foreground," "The black sedan in the middle lane," "The white minivan on the far left"). This is mandatory.
"""

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)

app = FastAPI(title="CarSpotter AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/identify-car")
async def identify_car(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await image.read()
        img = PIL.Image.open(io.BytesIO(image_bytes))
        response = model.generate_content(["Identify the car(s) in this image.", img], stream=False)
        
        if not response.parts:
             raise HTTPException(status_code=500, detail="Model returned an empty response.")
        
        return {"identification": response.text}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))