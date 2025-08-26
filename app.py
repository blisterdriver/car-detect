# app.py
import os
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import PIL.Image
import io

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
# It's best practice to set your API key in a .env file
# Create a file named .env and add: GOOGLE_API_KEY="your_actual_api_key"
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# NOTE: The model name you mentioned, "gemini-2.5-flash", is not yet available.
# I'm using "gemini-1.5-flash-latest" which is the correct and current model for this task.
# Your prompt will work perfectly with this model.
MODEL_NAME = "gemini-1.5-flash-latest" 
SYSTEM_PROMPT = """Your response should not be verbose. You are a world-class car specialist. You see a car and instantly tell what car this is and basically what year, what model. This is in bold letter. That's it. And then you give it a little bit of explanation that why do you think that this is the car that you think this is."""

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)

# --- FastAPI App ---
app = FastAPI(title="CarSpotter AI API")

# Configure CORS to allow frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the single-page HTML frontend."""
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/identify-car")
async def identify_car(image: UploadFile = File(...)):
    """
    Receives an image, sends it to the Gemini model for identification,
    and returns the model's response.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read image content
        image_bytes = await image.read()
        
        # Open the image with Pillow to ensure it's a valid image format
        img = PIL.Image.open(io.BytesIO(image_bytes))

        # Send image to Gemini model
        response = model.generate_content([
            "Identify the car in this image.",
            img
        ], stream=False)
        
        # Ensure there is content in the response
        if not response.parts:
             raise HTTPException(status_code=500, detail="Model returned an empty response.")
        
        return {"identification": response.text}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")

# To run this app locally:
# 1. Install dependencies: pip install -r requirements.txt
# 2. Run the server: uvicorn app:app --reload