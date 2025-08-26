# app.py
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

MODEL_NAME = "gemini-1.5-flash-latest"

# --- NEW, SPEC-FOCUSED SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a world-class automotive expert and data specialist. Your task is to identify all prominent cars in an image and provide their key performance specifications.

For each car identified, you MUST use the following exact Markdown format:

### **[Make Model (Estimated Year Range)]**
- **Location in Image:** [Clear, simple description, e.g., "The red SUV in the foreground"]
- **Engine:** [e.g., 2.0L Turbocharged I4, 3.5L V6]
- **Horsepower:** [e.g., 255 hp]
- **Torque:** [e.g., 273 lb-ft]
- **0-60 mph (0-100 km/h):** [e.g., ~5.9 seconds]
- **Top Speed:** [e.g., ~130 mph / 210 km/h]
- **Drivetrain:** [e.g., AWD, RWD, FWD]
- **Fuel Economy (MPG):** [e.g., ~22 City / 29 Hwy]

Provide the most common or representative specs for the identified model generation. Do not include any other commentary or reasoning.
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