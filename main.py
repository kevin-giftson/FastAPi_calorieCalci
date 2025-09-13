import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Calorie & Nutrition Analyzer API")

# ✅ CORS Middleware (handles OPTIONS preflight)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Simple test route (confirm CORS first)
@app.get("/ping")
async def ping():
    return {"msg": "pong ✅ CORS working"}

# Flexible response model
class NutritionResponse(BaseModel):
    nutrients: Dict[str, str]   # flexible key-value pairs
    food_tips: Optional[str]    # may or may not exist

@app.post("/analyze", response_model=NutritionResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload a food image and get nutrition values + AI-generated food tips.
    """
    try:
        # Save uploaded image temporarily
        image_path = f"temp_{file.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Gemini prompt
        prompt = """
        Analyze the food in this image.
        Return JSON ONLY in this exact structure:
        {
          "nutrients": {
            "carbs": "<number> g",
            "fat": "<number> g",
            "protein": "<number> g",
            "fiber": "<number> g",
            ...
          },
          "food_tips": "Write 2-3 sentences of advice about this meal."
        }
        Rules:
        - Include only nutrients you can identify (don’t force fixed keys).
        - All values must be strings ending with 'g', 'mg', or 'mcg'.
        - Do not include any text outside JSON.
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(
            [prompt, {"mime_type": file.content_type, "data": open(image_path, "rb").read()}]
        )

        raw_output = result.text.strip()

        # Clean markdown fences if Gemini adds them
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
        raw_output = raw_output.replace("json", "").strip()

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")

        return NutritionResponse(**parsed)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
