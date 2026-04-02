"""
Integration Module — Merge All Outputs
========================================
Merges all 5 modality CSVs into a single unified incident dataset.
Generates a final severity classification based on combined signals.

Steps (from assignment):
1. Define a common Incident_ID key across all five output CSVs.
2. Merge all five DataFrames using pandas.
3. Handle missing values where a modality has no data for a given incident.
4. Generate a final severity classification (Low / Medium / High) based on combined signals.
5. Output: output/final_integrated_report.csv
"""

import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.dirname(__file__)
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "final_integrated_report.csv")

# Expected CSV files from each modality (relative to integration folder)
CSV_FILES = {
    "audio": os.path.join("..", "audio", "audio_results.csv"),
    "pdf": os.path.join("..", "pdf", "pdf_results.csv"),
    "image": os.path.join("..", "images", "image_results.csv"),
    "video": os.path.join("..", "video", "video_results.csv"),
    "text": os.path.join("..", "text", "text_results.csv"),
}


def load_modality_csv(filepath):
    """Load a modality CSV if it exists and has data."""
    path = os.path.join(os.path.dirname(__file__), filepath)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            return df
        print(f"  WARNING: {filepath} is empty.")
    else:
        print(f"  WARNING: {filepath} not found. Skipping.")
    return None


def calculate_severity(row):
    """
    Calculate a final severity label based on combined signals from all modalities.
    
    Logic:
    - If urgency_score > 0.8 or severity_label is 'High'/'Critical' → 'High'
    - If urgency_score > 0.5 or any event is detected → 'Medium'
    - Otherwise → 'Low'
    """
    severity_signals = []

    # Check audio urgency
    if pd.notna(row.get("Urgency_Score")):
        try:
            urgency = float(row["Urgency_Score"])
            if urgency >= 0.8:
                severity_signals.append("High")
            elif urgency >= 0.5:
                severity_signals.append("Medium")
            else:
                severity_signals.append("Low")
        except (ValueError, TypeError):
            pass

    # Check text severity label
    if pd.notna(row.get("Severity_Label")):
        label = str(row["Severity_Label"]).strip()
        if label in ("High", "Critical"):
            severity_signals.append("High")
        elif label == "Medium":
            severity_signals.append("Medium")
        elif label == "Low":
            severity_signals.append("Low")

    # Check video/image confidence
    for col in ["Confidence_video", "Confidence_image"]:
        if pd.notna(row.get(col)):
            try:
                conf = float(row[col])
                if conf >= 0.85:
                    severity_signals.append("High")
                elif conf >= 0.6:
                    severity_signals.append("Medium")
            except (ValueError, TypeError):
                pass

    # Check if events indicate high severity
    event_cols = ["Extracted_Event", "Event_Detected", "Crime_Type"]
    high_severity_keywords = ["fire", "shooting", "weapon", "assault", "trapped",
                               "collapse", "explosion", "stabbing", "murder", "critical"]
    for col in event_cols:
        if pd.notna(row.get(col)):
            text = str(row[col]).lower()
            if any(kw in text for kw in high_severity_keywords):
                severity_signals.append("High")

    # Determine final severity
    if "High" in severity_signals:
        return "High"
    elif "Medium" in severity_signals:
        return "Medium"
    elif severity_signals:
        return "Low"
    else:
        return "Medium"  # Default when we have data but no strong signals


def merge_all():
    """
    Main merge function. Loads all available modality CSVs,
    assigns Incident_IDs, merges side-by-side, and calculates severity.
    
    Returns:
        pd.DataFrame: The final integrated report.
    """
    print("\n" + "=" * 60)
    print("  DATA INTEGRATION — Merging All Modality Outputs")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all available modality outputs
    modality_data = {}
    for modality, filename in CSV_FILES.items():
        df = load_modality_csv(filename)
        if df is not None:
            modality_data[modality] = df
            print(f"  ✓ Loaded {modality}: {len(df)} records from {filename}")

    if not modality_data:
        print("\n  ERROR: No modality data available to merge.")
        print("  Please run individual analyzers first.")
        return pd.DataFrame()

    # Determine the max number of rows across all modalities
    max_rows = max(len(df) for df in modality_data.values())
    print(f"\n  Max records across modalities: {max_rows}")
    print(f"  Generating {max_rows} incident record(s)...\n")

    # Build the integrated dataset
    integrated_rows = []
    for i in range(max_rows):
        incident_id = f"INC_{i + 1:03d}"
        row = {"Incident_ID": incident_id}

        # Audio columns
        if "audio" in modality_data and i < len(modality_data["audio"]):
            audio_row = modality_data["audio"].iloc[i]
            row["Audio_Event"] = audio_row.get("Extracted_Event", "")
            row["Audio_Sentiment"] = audio_row.get("Sentiment", "")
            row["Urgency_Score"] = audio_row.get("Urgency_Score", "")
        else:
            row["Audio_Event"] = ""
            row["Audio_Sentiment"] = ""
            row["Urgency_Score"] = ""

        # PDF columns
        if "pdf" in modality_data and i < len(modality_data["pdf"]):
            pdf_row = modality_data["pdf"].iloc[i]
            row["PDF_Doc_Type"] = pdf_row.get("Doc_Type", "")
            row["PDF_Summary"] = pdf_row.get("Summary", "")
        else:
            row["PDF_Doc_Type"] = ""
            row["PDF_Summary"] = ""

        # Image columns
        if "image" in modality_data and i < len(modality_data["image"]):
            img_row = modality_data["image"].iloc[i]
            row["Image_Objects"] = img_row.get("Objects_Detected", "")
            row["Image_Scene"] = img_row.get("Scene_Type", "")
            row["Confidence_image"] = img_row.get("Confidence", "")
        else:
            row["Image_Objects"] = ""
            row["Image_Scene"] = ""
            row["Confidence_image"] = ""

        # Video columns
        if "video" in modality_data and i < len(modality_data["video"]):
            vid_row = modality_data["video"].iloc[i]
            row["Video_Event"] = vid_row.get("Event_Detected", "")
            row["Confidence_video"] = vid_row.get("Confidence", "")
        else:
            row["Video_Event"] = ""
            row["Confidence_video"] = ""

        # Text columns
        if "text" in modality_data and i < len(modality_data["text"]):
            txt_row = modality_data["text"].iloc[i]
            row["Text_Crime_Type"] = txt_row.get("Crime_Type", "")
            row["Text_Topic"] = txt_row.get("Topic", "")
            row["Severity_Label"] = txt_row.get("Severity_Label", "")
        else:
            row["Text_Crime_Type"] = ""
            row["Text_Topic"] = ""
            row["Severity_Label"] = ""

        integrated_rows.append(row)

    # Create DataFrame and calculate severity
    df = pd.DataFrame(integrated_rows)
    df["Severity"] = df.apply(calculate_severity, axis=1)

    # Save final output
    df.to_csv(FINAL_OUTPUT, index=False)
    print(f"  Final integrated report saved to: {FINAL_OUTPUT}")
    print(f"  Total integrated records: {len(df)}")
    print(f"\n  Severity Distribution:")
    print(df["Severity"].value_counts().to_string())

    return df


def run():
    """Entry point for the integration module."""
    return merge_all()


if __name__ == "__main__":
    merge_all()
