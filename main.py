import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Calorie & Nutrition Analyzer API")

# Flexible response model
class NutritionResponse(BaseModel):
    nutrients: Dict[str, str]  # flexible key-value pairs like {"carbs": "30 g", "fat": "10 g"}
    food_tips: str             # AI-generated food advice

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

        # Send image to Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """
        Analyze the food in this image. 
        Return JSON in this exact structure:
        {
          "nutrients": {
             "carbs": "<number> g",
             "fat": "<number> g",
             "protein": "<number> g",
             "fiber": "<number> g",
             "iron": "<number> g",
             ...
          },
          "food_tips": "Write 2-3 sentences with advice about this meal. Mention what nutrients are strong or missing, and suggest improvements."
        }

        Rules:
        - Include only nutrients you can identify (don’t force fixed keys).
        - All values must be strings ending with 'g', 'mg', or 'mcg'.
        - Ensure valid JSON, no comments or extra text.
        """

        result = model.generate_content(
            [prompt, {"mime_type": file.content_type, "data": open(image_path, "rb").read()}]
        )

        # Extract raw model output
        raw_output = result.text.strip("```json").strip("```").strip()

        # Parse JSON safely
        parsed = json.loads(raw_output)

        # Return validated response
        return NutritionResponse(**parsed)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
