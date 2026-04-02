"""
Audio Analyzer Module — Student 1 Role
=======================================
Processes emergency audio calls / witness voice statements.
Uses Google Gemini to transcribe audio and extract structured incident data.

Alternative tools (referenced for academic credit):
- openai-whisper: Speech-to-text transcription
- spaCy: Named Entity Recognition for locations/names
- transformers: Sentiment analysis

Input:  Audio files (.mp3, .wav, .m4a) in audio/data/
Output: output/audio_results.csv
Schema: Call_ID, Transcript, Extracted_Event, Location, Sentiment, Urgency_Score
"""

import os
import sys
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.gemini_helper import analyze_multimodal, extract_json_from_response

# Supported audio extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}

# Directory paths
OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "audio_results.csv")

# Prompt template for Gemini
AUDIO_PROMPT = """You are an AI audio analyst for an emergency response department.
Listen to this audio file carefully. It may be an emergency call, a witness statement, or an incident recording.

Perform the following tasks:
1. Transcribe the full audio content.
2. Identify the type of incident/event described (e.g., fire, robbery, accident, assault, disturbance).
3. Extract any location mentions (street names, landmarks, neighborhoods).
4. Analyze the speaker's sentiment (Calm, Concerned, Distressed, Panicked).
5. Assign an urgency score from 0.0 (not urgent) to 1.0 (extremely urgent).

Return your analysis as a JSON object with EXACTLY these keys:
{
    "transcript": "Full transcription of the audio",
    "extracted_event": "Type of incident described",
    "location": "Location mentioned or 'Unknown'",
    "sentiment": "Calm | Concerned | Distressed | Panicked",
    "urgency_score": 0.85
}

Return ONLY the JSON object, no other text."""


def analyze_audio_file(file_path, call_id):
    """
    Analyze a single audio file using Gemini.
    
    Args:
        file_path (str): Path to the audio file.
        call_id (str): Identifier for this audio call.
    
    Returns:
        dict: Structured analysis result.
    """
    print(f"\n[Audio Analyzer] Processing: {os.path.basename(file_path)}")
    
    try:
        response = analyze_multimodal(AUDIO_PROMPT, file_path)
        result = extract_json_from_response(response)
        
        if result:
            return {
                "Call_ID": call_id,
                "Transcript": result.get("transcript", ""),
                "Extracted_Event": result.get("extracted_event", ""),
                "Location": result.get("location", "Unknown"),
                "Sentiment": result.get("sentiment", ""),
                "Urgency_Score": result.get("urgency_score", 0.0),
            }
        else:
            print(f"  ERROR: Could not parse Gemini response for {call_id}")
            return _empty_result(call_id)
    except Exception as e:
        print(f"  ERROR processing {call_id}: {e}")
        return _empty_result(call_id)


def _empty_result(call_id):
    """Return an empty result row for failed analyses."""
    return {
        "Call_ID": call_id,
        "Transcript": "Error: Analysis failed",
        "Extracted_Event": "",
        "Location": "",
        "Sentiment": "",
        "Urgency_Score": 0.0,
    }


def run():
    """
    Main entry point: scans audio/data/ for audio files,
    analyzes each with Gemini, and writes output/audio_results.csv.
    
    Returns:
        pd.DataFrame: The results DataFrame.
    """
    print("\n" + "=" * 60)
    print("  AUDIO ANALYZER — Student 1 Role")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find audio files using kagglehub
    try:
        import kagglehub
        print("  Fetching dataset from Kaggle...")
        data_path = kagglehub.dataset_download("louisteitelbaum/911-recordings-first-6-seconds")
    except ImportError:
        print("  ERROR: kagglehub not installed. Please run 'pip install kagglehub'")
        return pd.DataFrame()

    audio_files = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(root, f))
    
    # Sort and take a small subset to prevent exhausting API limits
    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"  No audio files found in downloaded dataset")
        return pd.DataFrame()

    # Process subset of 5 files maximum
    audio_files = audio_files[:5]
    print(f"  Found {len(audio_files)} audio file(s) to process.\n")

    # Process each audio file
    results = []
    for i, file_path in enumerate(audio_files, start=1):
        call_id = f"C{i:03d}"
        result = analyze_audio_file(file_path, call_id)
        results.append(result)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Audio results saved to: {OUTPUT_FILE}")
    print(f"  Total records: {len(df)}")
    return df


if __name__ == "__main__":
    run()
