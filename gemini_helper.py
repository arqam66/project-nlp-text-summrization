import os
import traceback
from dotenv import load_dotenv

load_dotenv(override=True)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

def generate_gemini_summary(text, num_sentences=3, api_key=None):
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY in .env or pass api_key.")

    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    client = genai.Client(api_key=key)

    prompt = f"""Summarize the following text in exactly {num_sentences} sentences. 
Return only the summary, no additional text.

Text:
{text}
"""
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}\n{traceback.format_exc()}")
