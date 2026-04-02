"""
Gemini API Helper Module
========================
Provides a shared interface to Google Gemini for all analyzer modules.
Handles API key loading, model configuration, structured JSON extraction,
and automatic retry with backoff for rate limit errors.
"""

import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please set it in the .env file.\n"
        "Example .env content:\n  GEMINI_API_KEY=your_api_key_here"
    )

genai.configure(api_key=GEMINI_API_KEY)

# Default model — gemini-2.5-flash is the latest stable model
DEFAULT_MODEL = "gemini-2.5-flash"

# Rate limit settings
REQUEST_DELAY = 5       # Seconds to wait between API calls
MAX_RETRIES = 3         # Max retry attempts on rate limit errors
RETRY_BASE_DELAY = 30   # Base delay (seconds) for exponential backoff


def get_gemini_model(model_name=None):
    """
    Returns a Gemini GenerativeModel instance.
    
    Args:
        model_name (str): The Gemini model to use. Defaults to DEFAULT_MODEL.
    Returns:
        genai.GenerativeModel
    """
    return genai.GenerativeModel(model_name or DEFAULT_MODEL)


def _call_with_retry(generate_fn, description="request"):
    """
    Wraps a Gemini API call with retry logic for rate limit (429) errors.
    Waits between requests to respect free-tier rate limits.
    
    Args:
        generate_fn: A callable that makes the Gemini API call.
        description: A label for logging.
    
    Returns:
        str: The model's text response.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Small delay between requests to avoid hitting rate limits
            if attempt == 0:
                time.sleep(REQUEST_DELAY)
            
            response = generate_fn()
            return response.text
        
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error (429)
            if "429" in error_str or "quota" in error_str.lower():
                wait_time = RETRY_BASE_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES:
                    print(f"  ⏳ Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  ❌ Rate limit exceeded after {MAX_RETRIES} retries.")
                    raise
            else:
                # Non-rate-limit error, raise immediately
                raise


def analyze_with_gemini(prompt, model_name=None):
    """
    Send a text-only prompt to Gemini and return the response text.
    Includes automatic retry for rate limit errors.
    
    Args:
        prompt (str): The prompt to send.
        model_name (str): Which Gemini model to use.
    
    Returns:
        str: The model's text response.
    """
    model = get_gemini_model(model_name)
    return _call_with_retry(
        lambda: model.generate_content(prompt),
        description="text analysis"
    )


def analyze_multimodal(prompt, file_path, model_name=None):
    """
    Send a multimodal prompt (text + file) to Gemini.
    Supports images, audio, video, and PDF files.
    Includes automatic retry for rate limit errors.
    
    Args:
        prompt (str): The text instruction to accompany the file.
        file_path (str): Absolute path to the media file.
        model_name (str): Which Gemini model to use.
    
    Returns:
        str: The model's text response.
    """
    model = get_gemini_model(model_name)

    # Upload the file using Gemini File API
    print(f"  Uploading file: {os.path.basename(file_path)}...")
    uploaded_file = genai.upload_file(file_path)
    print(f"  File uploaded successfully. Generating analysis...")

    return _call_with_retry(
        lambda: model.generate_content([prompt, uploaded_file]),
        description="multimodal analysis"
    )


def extract_json_from_response(response_text):
    """
    Extracts a JSON object or array from a Gemini response string.
    Handles cases where Gemini wraps JSON in markdown code blocks.
    
    Args:
        response_text (str): Raw text from Gemini.
    
    Returns:
        dict or list: Parsed JSON data.
    """
    text = response_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON within the text
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            try:
                return json.loads(text[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass

        # Try array
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            try:
                return json.loads(text[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass

        print(f"  WARNING: Could not parse JSON from response. Raw text:\n{text[:500]}")
        return None
