"""
Video Analyzer Module — Student 4 Role
=======================================
Processes CCTV / surveillance footage to detect events and activities.
Uses OpenCV for frame extraction and Google Gemini Vision for analysis.

Alternative tools (referenced for academic credit):
- YOLOv8: Real-time object detection on extracted frames
- PyTorch / TensorFlow: Anomaly detection model implementation
- imageio / moviepy: Video loading and manipulation

Input:  Video files (.mp4, .avi, .mov, .mpg, .mpeg) in video/data/
Output: output/video_results.csv
Schema: Clip_ID, Timestamp, Frame_ID, Event_Detected, Persons_Count, Confidence
"""

import os
import sys
import math
import pandas as pd
import cv2

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.gemini_helper import analyze_multimodal, extract_json_from_response

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mpg", ".mpeg", ".mkv", ".wmv"}

# Directory paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "video_results.csv")
FRAMES_DIR = os.path.join(os.path.dirname(__file__), "extracted_frames")

# How many frames to extract per video clip
FRAMES_PER_VIDEO = 5

# Prompt for Gemini video frame analysis
FRAME_PROMPT = """You are an AI video surveillance analyst reviewing a frame extracted from CCTV footage.
Analyze this frame as if it is evidence from a security camera.

Perform the following tasks:
1. Detect any events or activities of interest (e.g., person walking, running, fighting, collapsing, vehicle movement, fire, theft, loitering).
2. Count the number of persons visible in the frame.
3. Identify any objects of interest (vehicles, bags, weapons, fire).
4. Estimate a confidence score from 0.0 to 1.0 for your event detection.

Return your analysis as a JSON object with EXACTLY these keys:
{
    "event_detected": "Description of event or activity",
    "persons_count": "Number of persons visible (e.g., '2 persons')",
    "objects": "Objects of interest detected",
    "confidence": 0.85
}

Return ONLY the JSON object, no other text."""

# Prompt for full video analysis via Gemini
VIDEO_PROMPT = """You are an AI video surveillance analyst for a law enforcement agency.
Watch this video clip carefully. It may be CCTV footage, dashcam video, or security camera recording.

Analyze the full video and identify ALL distinct events or activities that occur.
For each event, provide:
1. The approximate timestamp when it occurs.
2. Description of the event (e.g., person walking, running, fighting, collapsing, vehicle movement, fire).
3. Number of persons visible during the event.
4. A confidence score from 0.0 to 1.0.

Return your analysis as a JSON ARRAY where each element represents one detected event:
[
    {
        "timestamp": "00:00:05",
        "event_detected": "Person walking through corridor",
        "persons_count": "1 person",
        "confidence": 0.90
    },
    {
        "timestamp": "00:00:12",
        "event_detected": "Two people meeting and talking",
        "persons_count": "2 persons",
        "confidence": 0.85
    }
]

If no notable events are detected, return a single-element array with event_detected as "No notable activity".

Return ONLY the JSON array, no other text."""


def extract_frames(video_path, clip_id, num_frames=FRAMES_PER_VIDEO):
    """
    Extract evenly-spaced frames from a video using OpenCV.
    
    Args:
        video_path (str): Path to the video file.
        clip_id (str): Identifier for this clip.
        num_frames (int): Number of frames to extract.
    
    Returns:
        list of tuples: [(frame_path, timestamp_str, frame_id), ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25.0  # Default fallback
    duration = total_frames / fps

    print(f"  Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")

    # Calculate frame indices to extract
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / num_frames
        frame_indices = [int(step * i) for i in range(num_frames)]

    # Create frames directory
    clip_frames_dir = os.path.join(FRAMES_DIR, clip_id)
    os.makedirs(clip_frames_dir, exist_ok=True)

    frames = []
    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            timestamp_sec = frame_num / fps
            minutes = int(timestamp_sec // 60)
            seconds = int(timestamp_sec % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
            frame_id = f"FRM_{idx + 1:03d}"

            frame_path = os.path.join(clip_frames_dir, f"{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append((frame_path, timestamp_str, frame_id))

    cap.release()
    print(f"  Extracted {len(frames)} frames to: {clip_frames_dir}")
    return frames


def analyze_video_full(file_path, clip_id):
    """
    Analyze a full video file by uploading it directly to Gemini.
    This is the preferred approach — faster and more accurate.
    Falls back to frame-by-frame analysis if the upload fails.
    
    Args:
        file_path (str): Path to the video file.
        clip_id (str): Identifier for this video clip.
    
    Returns:
        list of dict: List of event records.
    """
    print(f"\n[Video Analyzer] Processing: {os.path.basename(file_path)}")

    results = []

    try:
        # Try full video upload to Gemini first
        print("  Attempting full video analysis via Gemini...")
        response = analyze_multimodal(VIDEO_PROMPT, file_path)
        events = extract_json_from_response(response)

        if events and isinstance(events, list):
            for idx, event in enumerate(events):
                results.append({
                    "Clip_ID": clip_id,
                    "Timestamp": event.get("timestamp", "00:00"),
                    "Frame_ID": f"FRM_{idx + 1:03d}",
                    "Event_Detected": event.get("event_detected", ""),
                    "Persons_Count": event.get("persons_count", "0"),
                    "Confidence": event.get("confidence", 0.0),
                })
            return results
        elif events and isinstance(events, dict):
            results.append({
                "Clip_ID": clip_id,
                "Timestamp": events.get("timestamp", "00:00"),
                "Frame_ID": "FRM_001",
                "Event_Detected": events.get("event_detected", ""),
                "Persons_Count": events.get("persons_count", "0"),
                "Confidence": events.get("confidence", 0.0),
            })
            return results

    except Exception as e:
        print(f"  Full video upload failed: {e}")
        print("  Falling back to frame-by-frame analysis...")

    # Fallback: Extract frames and analyze individually
    try:
        frames = extract_frames(file_path, clip_id)
        if not frames:
            return [_empty_result(clip_id)]

        for frame_path, timestamp_str, frame_id in frames:
            try:
                response = analyze_multimodal(FRAME_PROMPT, frame_path)
                result = extract_json_from_response(response)
                if result:
                    results.append({
                        "Clip_ID": clip_id,
                        "Timestamp": timestamp_str,
                        "Frame_ID": frame_id,
                        "Event_Detected": result.get("event_detected", ""),
                        "Persons_Count": result.get("persons_count", "0"),
                        "Confidence": result.get("confidence", 0.0),
                    })
                else:
                    results.append(_empty_result(clip_id, timestamp_str, frame_id))
            except Exception as e2:
                print(f"  ERROR on frame {frame_id}: {e2}")
                results.append(_empty_result(clip_id, timestamp_str, frame_id))

    except Exception as e:
        print(f"  ERROR processing {clip_id}: {e}")
        results.append(_empty_result(clip_id))

    return results if results else [_empty_result(clip_id)]


def _empty_result(clip_id, timestamp="00:00", frame_id="FRM_001"):
    """Return an empty result row for failed analyses."""
    return {
        "Clip_ID": clip_id,
        "Timestamp": timestamp,
        "Frame_ID": frame_id,
        "Event_Detected": "Error: Analysis failed",
        "Persons_Count": "0",
        "Confidence": 0.0,
    }


def run():
    """
    Main entry point: scans video/data/ for video files,
    analyzes each, and writes output/video_results.csv.
    
    Returns:
        pd.DataFrame: The results DataFrame.
    """
    print("\n" + "=" * 60)
    print("  VIDEO ANALYZER — Student 4 Role")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"  No data directory found. Created: {DATA_DIR}")
        print(f"  Please place video files (.mp4, .mpg) in {DATA_DIR}")
        return pd.DataFrame()

    video_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ])

    if not video_files:
        print(f"  No video files found in {DATA_DIR}")
        print(f"  Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
        return pd.DataFrame()

    # Limit number of videos to avoid excessive API calls on free tier
    MAX_VIDEOS = 5
    if len(video_files) > MAX_VIDEOS:
        print(f"  Found {len(video_files)} video files. Processing first {MAX_VIDEOS} to stay within API limits.")
        video_files = video_files[:MAX_VIDEOS]
    else:
        print(f"  Found {len(video_files)} video file(s) to process.")

    all_results = []
    for i, filename in enumerate(video_files, start=1):
        clip_id = f"CLIP_{i:03d}"
        file_path = os.path.join(DATA_DIR, filename)
        results = analyze_video_full(file_path, clip_id)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Video results saved to: {OUTPUT_FILE}")
    print(f"  Total records: {len(df)}")
    return df


if __name__ == "__main__":
    run()
