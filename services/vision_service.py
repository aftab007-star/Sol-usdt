import os
from google import genai
from google.genai import types
from PIL import Image
import io
import re
from database.db_manager import log_sentiment

# Initialize the NEW client
# It automatically picks up GEMINI_API_KEY from your .env
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"

def get_sentiment(image_bytes):
    """Analyzes an image using the new Google Gen AI SDK."""
    try:
        # Load image for processing
        image = Image.open(io.BytesIO(image_bytes))
        
        # New SDK syntax for multimodal generation
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                "You are a crypto analyst. Analyze this news screenshot. "
                "Provide a sentiment verdict (BULLISH, BEARISH, or NEUTRAL) "
                "and a confidence score (0-100). Format: Verdict: [VERDICT], Confidence: [SCORE].",
                image
            ]
        )
        
        text_response = response.text
        
        # Parse verdict and confidence
        verdict_match = re.search(r"Verdict: (\w+)", text_response, re.IGNORECASE)
        confidence_match = re.search(r"Confidence: (\d+)", text_response, re.IGNORECASE)
        
        verdict = verdict_match.group(1).upper() if verdict_match else "NEUTRAL"
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0
        
        # Log to database
        log_sentiment(verdict, confidence)
        
        return text_response
    except Exception as e:
        print(f"Vision Error: {e}")
        # Log error case
        log_sentiment("NEUTRAL", 0.0)
        return "NEUTRAL (Error in analysis)"

# Backward compatible alias for callers expecting the old name
analyze_image = get_sentiment
