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

MODEL_NAME = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = """Your primary and most important task is to identify as many cars as possible in the image, even if they are partially obscured, far away, or not the main subject. Be thorough.

For each car you identify, you MUST use the following exact Markdown format:

### **[Make Model (Estimated Year Range)]**
- **Location in Image:** [Clear, simple description, e.g., "The red SUV in the foreground", "The silver minivan on the far left"]
- **Engine:** [e.g., 2.0L Turbocharged I4, 3.5L V6]
- **Horsepower:** [e.g., 255 hp]
- **Torque:** [e.g., 273 lb-ft]
- **0-60 mph (0-100 km/h):** [e.g., ~5.9 seconds]
- **Top Speed:** [e.g., ~130 mph / 210 km/h]
- **Drivetrain:** [e.g., AWD, RWD, FWD]
- **Fuel Economy (MPG):** [e.g., ~22 City / 29 Hwy]
- Country of origin. 

CRITICAL FALLBACK RULE: If you cannot confidently identify a vehicle due to poor image quality (blurry, too distant, obscured), you MUST format the title as:
`### **Unspecified Car (Reason for uncertainty)**`
The reason must be concise, like `(Too Blurry)`, `(Partially Obscured)`, or `(Not Enough Confidence)`. Provide specs for unspecified cars be accurate as much as possible.
"""

generation_config = {
  "temperature": 0.5,
}

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    MODEL_NAME, 
    system_instruction=SYSTEM_PROMPT,
    generation_config=generation_config
)

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
    # --- THIS IS THE FIX ---
    # Explicitly open the file with UTF-8 encoding to prevent the server crash,
    # especially on Windows environments.
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found.</h1>", status_code=404)


@app.post("/identify-car")
async def identify_car(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await image.read()
        img = PIL.Image.open(io.BytesIO(image_bytes))
        response = model.generate_content(["Identify the car(s) in this image.", img], stream=False)
        
        if not response.parts:
             raise HTTPException(status_code=500, detail="The AI model returned an empty response. Please try a different image.")
        
        return {"identification": response.text}
    except Exception as e:
        print(f"An internal AI model error occurred: {e}")
        raise HTTPException(status_code=503, detail="Too much overload on this site. Please slow down.")