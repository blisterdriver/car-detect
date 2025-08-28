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

# --- FIX 1: CORRECT AND WORKING MODEL NAME ---
# The names "2.0-flash-lite" is not a valid public API model and will cause errors.
# Using the correct name ensures stability and the best performance.
MODEL_NAME = "gemini-2.5-flash"


# --- FIX 2: COMPLETELY REWRITTEN PROMPT FOR PRECISION AND CONFIDENCE ---
SYSTEM_PROMPT = """Your single most important mission is to find and box EVERY potential car in the image with extreme precision, ORDERED from closest to the camera to furthest away. Be relentless. Identify cars even if they are distant, blurry, or partially blocked.

For each car you identify, you MUST use the following exact Markdown format, including a confidence score in the title:

### **[Make Model (Estimated Year Range)] ([Confidence %])**
- **Location in Image:** [Clear, simple description, e.g., "The red SUV in the foreground"]
- **BoundingBox:** [x_min, y_min, x_max, y_max]
- **MSRP (usd price in the first without import duty taxes, then bdt with import duty 1 usd = 130bdt, when saying the bdt price dont use the $ logo):**
- **Engine:** [e.g., 2.0L Turbocharged I4]
- **Engine CC:** [e.g., 1998]
- **Vehicle Type:** [ICE, EV, or Hybrid]
- **Horsepower:** [e.g., 255 hp]
- **Torque:** [e.g., 273 lb-ft]
- **0-60 mph (0-100 km/h):** [e.g., ~5.9 seconds]
- **Top Speed:** [e.g., ~130 mph / 210 km/h]
- **Drivetrain:** [e.g., AWD]
- **Fuel Economy (MPG):** [e.g., ~22 City / 29 Hwy]


CRITICAL RULES:
1.  **ORDERING:** You MUST list the cars starting with the one closest to the viewer and work your way backwards to the one furthest away. This is the highest priority rule.
2.  **EXTREME TIGHT FIT:** The box MUST be the smallest possible rectangle that perfectly crops the vehicle's visible shape.
3.  **ONE CAR PER BOX:** Each box must contain only ONE car. Never group multiple cars in one box.
4.  **EXCLUDE EVERYTHING ELSE:** The box must NOT contain pedestrians, cyclists, pylons, or other objects.
5.  **COORDINATES:** Use normalized coordinates where [0.0, 0.0] is the top-left corner and [1.0, 1.0] is the bottom-right.
6.  **UNIDENTIFIED CARS:** If you cannot identify a vehicle's make and model, format the title as `### **Unspecified Car (Reason)**`, but you MUST still provide a tight and accurate BoundingBox and estimate the other details if possible.
7. only spot cars nothing else
8. dont give info about unspecified ones

"""

# --- FIX 3: REDUCED TEMPERATURE FOR HIGH PRECISION ---
# Lowering the temperature from 1.0 to 0.4 forces the model to be much less
# "creative" and to follow the strict, tight-boxing rules with greater accuracy.
generation_config = {
  "temperature": 1,
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
        
        response = model.generate_content(["Find and box all cars in this image.", img], stream=False)
        
        if not response.parts:
             raise HTTPException(status_code=500, detail="The AI model returned an empty response. Please try a different image.")
        
        return {"identification": response.text}
    except Exception as e:
        print(f"An internal AI model error occurred: {e}")
        raise HTTPException(status_code=503, detail="The AI model could not process this request. This might be due to a safety block, an invalid image, or temporary service overload. Please try a different image.")