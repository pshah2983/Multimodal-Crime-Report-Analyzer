"""
Text Analyzer Module — Student 5 Role
=======================================
Processes social media posts, news articles, and crime text reports.
Uses Google Gemini for NER, sentiment analysis, and topic classification.

Alternative tools (referenced for academic credit):
- spaCy: NER and text preprocessing
- transformers (HuggingFace): Sentiment analysis and zero-shot topic classification
- NLTK: Tokenization, stopword removal, stemming
- pandas: Structured output generation

Input:  Text files (.txt) or CSV files (.csv) in text/data/
Output: output/text_results.csv
Schema: Text_ID, Crime_Type, Location_Entity, Sentiment, Topic, Severity_Label
"""

import os
import sys
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.gemini_helper import analyze_with_gemini, extract_json_from_response

# Directory paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "text_results.csv")

# Prompt for single text analysis
TEXT_PROMPT = """You are an AI text analyst for a law enforcement intelligence unit.
Analyze the following text which may be from a crime report, social media post, news article, or witness statement.

Perform the following tasks:
1. Identify the crime or incident type (e.g., Robbery, Assault, Fire, Traffic Accident, Vandalism, Theft, Disturbance).
2. Extract location entities mentioned (street names, cities, landmarks). If none, say "Unknown".
3. Determine the overall sentiment (Positive, Neutral, Negative, Very Negative).
4. Classify the topic (e.g., Theft / Robbery, Violence / Assault, Fire / Hazard, Traffic / Accident, Public Disturbance, Drug-related, Other).
5. Assign a severity label (Low, Medium, High, Critical) based on the nature and urgency of the incident.

Return your analysis as a JSON object with EXACTLY these keys:
{
    "crime_type": "Robbery",
    "location_entity": "Oak Street, Chicago",
    "sentiment": "Negative",
    "topic": "Theft / Robbery",
    "severity_label": "High"
}

Return ONLY the JSON object, no other text.

--- TEXT TO ANALYZE ---
"""

# Prompt for batch text analysis (multiple rows from a CSV)
BATCH_TEXT_PROMPT = """You are an AI text analyst for a law enforcement intelligence unit.
Analyze EACH of the following crime report texts individually.

For each text, extract:
1. Crime type (Robbery, Assault, Fire, Traffic Accident, Vandalism, etc.)
2. Location entities (streets, cities, landmarks or "Unknown")
3. Sentiment (Positive, Neutral, Negative, Very Negative)
4. Topic classification (Theft/Robbery, Violence/Assault, Fire/Hazard, Traffic/Accident, etc.)
5. Severity label (Low, Medium, High, Critical)

Return a JSON ARRAY where each element corresponds to one input text, in order:
[
    {
        "crime_type": "Robbery",
        "location_entity": "Oak Street",
        "sentiment": "Negative",
        "topic": "Theft / Robbery",
        "severity_label": "High"
    }
]

Return ONLY the JSON array, no other text.

--- TEXTS TO ANALYZE ---
"""


def analyze_single_text(text, text_id):
    """
    Analyze a single text string using Gemini.
    
    Args:
        text (str): The text content to analyze.
        text_id (str): Identifier for this text.
    
    Returns:
        dict: Structured analysis result.
    """
    try:
        prompt = TEXT_PROMPT + text[:3000]  # Truncate very long texts
        response = analyze_with_gemini(prompt)
        result = extract_json_from_response(response)

        if result:
            return {
                "Text_ID": text_id,
                "Crime_Type": result.get("crime_type", ""),
                "Location_Entity": result.get("location_entity", "Unknown"),
                "Sentiment": result.get("sentiment", ""),
                "Topic": result.get("topic", ""),
                "Severity_Label": result.get("severity_label", ""),
            }
        else:
            return _empty_result(text_id)

    except Exception as e:
        print(f"  ERROR processing {text_id}: {e}")
        return _empty_result(text_id)


def analyze_csv_file(file_path):
    """
    Process a CSV file containing crime text reports.
    Looks for columns like 'text', 'description', 'report', 'narrative', or 'crimeaditionalinfo'.
    Analyzes texts in batches for efficiency.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        list of dict: Analysis results for each row.
    """
    print(f"\n[Text Analyzer] Processing CSV: {os.path.basename(file_path)}")

    try:
        df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")
        except Exception as e:
            print(f"  ERROR reading CSV: {e}")
            return []

    print(f"  CSV shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Find the text column — try common names
    text_col = None
    possible_names = ["text", "description", "report", "narrative", "content",
                      "crimeaditionalinfo", "crime_description", "details",
                      "summary", "body", "message"]
    for col in df.columns:
        if col.strip().lower() in possible_names:
            text_col = col
            break

    if text_col is None:
        # Fall back to the column with the longest average text
        text_cols = df.select_dtypes(include=["object"]).columns
        if len(text_cols) > 0:
            avg_lens = {c: df[c].astype(str).str.len().mean() for c in text_cols}
            text_col = max(avg_lens, key=avg_lens.get)
            print(f"  Auto-selected text column: '{text_col}' (avg length: {avg_lens[text_col]:.0f})")
        else:
            print("  ERROR: No suitable text column found in CSV.")
            return []
    else:
        print(f"  Using text column: '{text_col}'")

    # Limit to first 20 rows to avoid excessive API calls
    sample_df = df.head(20).copy()
    sample_df[text_col] = sample_df[text_col].astype(str).fillna("")

    results = []
    for idx, row in sample_df.iterrows():
        text = row[text_col]
        if not text or text.lower() == "nan" or len(text.strip()) < 10:
            continue

        text_id = f"TXT_{len(results) + 1:03d}"
        print(f"  Analyzing {text_id}: {text[:80]}...")
        result = analyze_single_text(text, text_id)
        results.append(result)

    return results


def analyze_json_tweets_file(file_path):
    """
    Process a .txt file containing JSON tweets (one JSON object per line).
    Extracts the 'text' field from each tweet for crime analysis.
    Handles the CrimeReport dataset format from Kaggle.
    
    Args:
        file_path (str): Path to the JSON-per-line file.
    
    Returns:
        list of dict: Analysis results for each tweet.
    """
    import json as json_module

    print(f"\n[Text Analyzer] Processing JSON tweets: {os.path.basename(file_path)}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            lines = f.readlines()

    # Extract tweet texts from JSON lines
    tweets = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json_module.loads(line)
            # The tweet text is in the 'text' field
            text = data.get("text", "")
            if text and len(text.strip()) > 15:
                tweets.append(text)
        except json_module.JSONDecodeError:
            continue

    print(f"  Parsed {len(tweets)} tweets with text content.")

    # Limit to 15 tweets to avoid excessive API calls
    MAX_TWEETS = 15
    tweets = tweets[:MAX_TWEETS]
    print(f"  Analyzing {len(tweets)} tweet(s) (max {MAX_TWEETS})...\n")

    results = []
    for i, text in enumerate(tweets, start=1):
        text_id = f"TXT_{i:03d}"
        print(f"  Analyzing {text_id}: {text[:80]}...")
        result = analyze_single_text(text, text_id)
        results.append(result)

    return results


def analyze_txt_file(file_path, text_id):
    """
    Process a single .txt file containing one crime report.
    Auto-detects whether the file is JSON-per-line (tweets) or plain text.
    
    Args:
        file_path (str): Path to the text file.
        text_id (str): Identifier for this text.
    
    Returns:
        dict or list: Structured analysis result(s).
    """
    print(f"\n[Text Analyzer] Processing: {os.path.basename(file_path)}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            first_line = f.readline().strip()

    # Detect if it's a JSON-per-line file (tweet format)
    if first_line.startswith("{") and '"text"' in first_line:
        print("  Detected JSON-per-line tweet format.")
        return analyze_json_tweets_file(file_path)

    # Otherwise treat as plain text
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()

    if not text.strip():
        print(f"  WARNING: Empty text file: {file_path}")
        return _empty_result(text_id)

    return analyze_single_text(text, text_id)


def _empty_result(text_id):
    """Return an empty result row for failed analyses."""
    return {
        "Text_ID": text_id,
        "Crime_Type": "Error: Analysis failed",
        "Location_Entity": "",
        "Sentiment": "",
        "Topic": "",
        "Severity_Label": "",
    }


def run():
    """
    Main entry point: scans text/data/ for text and CSV files,
    analyzes each, and writes output/text_results.csv.
    
    Returns:
        pd.DataFrame: The results DataFrame.
    """
    print("\n" + "=" * 60)
    print("  TEXT ANALYZER — Student 5 Role")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"  No data directory found. Created: {DATA_DIR}")
        print(f"  Please place text files (.txt, .csv) in {DATA_DIR}")
        return pd.DataFrame()

    # Find all text and CSV files
    txt_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".txt")])
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")])

    if not txt_files and not csv_files:
        print(f"  No text or CSV files found in {DATA_DIR}")
        return pd.DataFrame()

    all_results = []

    # Process CSV files (e.g., Kaggle CrimeReport dataset)
    for filename in csv_files:
        file_path = os.path.join(DATA_DIR, filename)
        csv_results = analyze_csv_file(file_path)
        all_results.extend(csv_results)

    # Process individual .txt files
    for filename in txt_files:
        text_id = f"TXT_{len(all_results) + 1:03d}"
        file_path = os.path.join(DATA_DIR, filename)
        result = analyze_txt_file(file_path, text_id)
        # analyze_txt_file can return a list (JSON tweets) or a single dict
        if isinstance(result, list):
            all_results.extend(result)
        else:
            all_results.append(result)

    if not all_results:
        print("  No results generated.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Text results saved to: {OUTPUT_FILE}")
    print(f"  Total records: {len(df)}")
    return df


if __name__ == "__main__":
    run()

