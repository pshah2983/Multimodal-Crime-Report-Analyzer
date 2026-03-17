"""
Image Analyzer Module — Student 3 Role
=======================================
Processes crime scene / accident scene photographs.
Uses Google Gemini Vision to detect objects, classify scenes, and perform OCR.

Alternative tools (referenced for academic credit):
- YOLOv8 (ultralytics): State-of-the-art object detection
- OpenCV: Image preprocessing and frame handling
- pytesseract: OCR for text visible in images
- torchvision / HuggingFace: Pre-trained image classification models

Input:  Image files (.jpg, .jpeg, .png, .bmp) in images/data/
Output: output/image_results.csv
Schema: Image_ID, Scene_Type, Objects_Detected, Bounding_Boxes, Confidence
"""

import os
import sys
import pandas as pd
from PIL import Image

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.gemini_helper import analyze_multimodal, extract_json_from_response

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Directory paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "image_results.csv")

# Prompt for Gemini image analysis
IMAGE_PROMPT = """You are an AI image analyst for a law enforcement crime scene investigation unit.
Analyze this image carefully as if it were evidence from a crime or incident scene.

Perform the following tasks:
1. Classify the scene type (e.g., Fire Scene, Traffic Accident, Theft, Vandalism, Assault, Public Disturbance, Normal).
2. Detect and list all relevant objects visible (e.g., vehicles, fire, smoke, weapons, people, damage, debris).
3. Describe approximate bounding box regions for detected objects (e.g., "2 fire regions in center", "1 vehicle on left side").
4. Estimate a confidence score from 0.0 to 1.0 for your overall scene classification.
5. Extract any visible text in the image using OCR (license plates, street signs, labels).

Return your analysis as a JSON object with EXACTLY these keys:
{
    "scene_type": "Fire Scene",
    "objects_detected": "fire, smoke, building, firefighter",
    "bounding_boxes": "2 fire regions center-right, 1 smoke plume top, 1 person left",
    "confidence": 0.92,
    "text_extracted": "Any visible text or 'None'"
}

Return ONLY the JSON object, no other text."""


def analyze_image_file(file_path, image_id):
    """
    Analyze a single image file using Gemini Vision.
    
    Args:
        file_path (str): Path to the image file.
        image_id (str): Identifier for this image.
    
    Returns:
        dict: Structured analysis result.
    """
    print(f"\n[Image Analyzer] Processing: {os.path.basename(file_path)}")

    try:
        # Verify the image is valid
        img = Image.open(file_path)
        print(f"  Image size: {img.size}, Mode: {img.mode}")

        response = analyze_multimodal(IMAGE_PROMPT, file_path)
        result = extract_json_from_response(response)

        if result:
            return {
                "Image_ID": image_id,
                "Scene_Type": result.get("scene_type", ""),
                "Objects_Detected": result.get("objects_detected", ""),
                "Bounding_Boxes": result.get("bounding_boxes", ""),
                "Confidence": result.get("confidence", 0.0),
            }
        else:
            print(f"  ERROR: Could not parse Gemini response for {image_id}")
            return _empty_result(image_id)

    except Exception as e:
        print(f"  ERROR processing {image_id}: {e}")
        return _empty_result(image_id)


def _empty_result(image_id):
    """Return an empty result row for failed analyses."""
    return {
        "Image_ID": image_id,
        "Scene_Type": "Error: Analysis failed",
        "Objects_Detected": "",
        "Bounding_Boxes": "",
        "Confidence": 0.0,
    }


def find_images_recursive(directory, max_images=10):
    """
    Recursively search for image files in a directory and its subdirectories.
    Handles Roboflow YOLOv8 dataset structure (train/images/, valid/images/, etc.)
    
    Args:
        directory (str): Root directory to search.
        max_images (int): Maximum number of images to return.
    
    Returns:
        list: List of full file paths to images.
    """
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, f))
                if len(image_paths) >= max_images:
                    return image_paths
    return image_paths


def run():
    """
    Main entry point: scans images/data/ (including subdirectories) for image files,
    analyzes each with Gemini Vision, and writes output/image_results.csv.
    Supports Roboflow YOLOv8 dataset structure (train/images/, valid/images/, etc.)
    Limits processing to 10 images to avoid excessive API calls.
    
    Returns:
        pd.DataFrame: The results DataFrame.
    """
    print("\n" + "=" * 60)
    print("  IMAGE ANALYZER — Student 3 Role")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"  No data directory found. Created: {DATA_DIR}")
        print(f"  Please place image files (.jpg, .png) in {DATA_DIR}")
        return pd.DataFrame()

    # Recursively find images (handles Roboflow YOLOv8 folder structure)
    MAX_IMAGES = 10  # Limit to avoid excessive API calls
    image_paths = find_images_recursive(DATA_DIR, max_images=MAX_IMAGES)

    if not image_paths:
        print(f"  No image files found in {DATA_DIR} or its subdirectories")
        print(f"  Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        return pd.DataFrame()

    print(f"  Found image files. Processing {len(image_paths)} image(s) (max {MAX_IMAGES}).\n")

    results = []
    for i, file_path in enumerate(image_paths, start=1):
        image_id = f"IMG_{i:03d}"
        result = analyze_image_file(file_path, image_id)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Image results saved to: {OUTPUT_FILE}")
    print(f"  Total records: {len(df)}")
    return df


if __name__ == "__main__":
    run()
