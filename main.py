"""
Main Orchestrator — Multimodal Crime / Incident Report Analyzer
=================================================================
Entry point that runs all 5 modality analyzers in sequence,
then merges results into a final integrated incident report.

Usage:
    python main.py              # Run all analyzers + integration
    python main.py --audio      # Run only audio analyzer
    python main.py --pdf        # Run only PDF analyzer
    python main.py --image      # Run only image analyzer
    python main.py --video      # Run only video analyzer
    python main.py --text       # Run only text analyzer
    python main.py --merge      # Run only the integration/merge step
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def run_audio():
    """Run the Audio Analyzer (Student 1)."""
    from audio.audio_analyzer import run
    return run()


def run_pdf():
    """Run the PDF Analyzer (Student 2)."""
    from pdf.pdf_analyzer import run
    return run()


def run_image():
    """Run the Image Analyzer (Student 3)."""
    from images.image_analyzer import run
    return run()


def run_video():
    """Run the Video Analyzer (Student 4)."""
    from video.video_analyzer import run
    return run()


def run_text():
    """Run the Text Analyzer (Student 5)."""
    from text.text_analyzer import run
    return run()


def run_merge():
    """Run the Integration / Merge step."""
    from integration.merge import run
    return run()


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Crime / Incident Report Analyzer"
    )
    parser.add_argument("--audio", action="store_true", help="Run only Audio Analyzer")
    parser.add_argument("--pdf", action="store_true", help="Run only PDF Analyzer")
    parser.add_argument("--image", action="store_true", help="Run only Image Analyzer")
    parser.add_argument("--video", action="store_true", help="Run only Video Analyzer")
    parser.add_argument("--text", action="store_true", help="Run only Text Analyzer")
    parser.add_argument("--merge", action="store_true", help="Run only the merge step")
    args = parser.parse_args()

    # If a specific flag is set, run only that module
    specific = args.audio or args.pdf or args.image or args.video or args.text or args.merge
    if specific:
        if args.audio:
            run_audio()
        if args.pdf:
            run_pdf()
        if args.image:
            run_image()
        if args.video:
            run_video()
        if args.text:
            run_text()
        if args.merge:
            run_merge()
        return

    # Default: run the full pipeline
    print("\n" + "#" * 60)
    print("#  MULTIMODAL CRIME / INCIDENT REPORT ANALYZER")
    print("#  Full Pipeline Execution")
    print("#" * 60)

    analyzer_counts = {}

    # Step 1: Audio Analysis
    try:
        df = run_audio()
        analyzer_counts["Audio"] = len(df) if df is not None and not df.empty else 0
    except Exception as e:
        print(f"\n  ⚠ Audio Analyzer error: {e}")
        analyzer_counts["Audio"] = 0

    # Step 2: PDF Analysis
    try:
        df = run_pdf()
        analyzer_counts["PDF"] = len(df) if df is not None and not df.empty else 0
    except Exception as e:
        print(f"\n  ⚠ PDF Analyzer error: {e}")
        analyzer_counts["PDF"] = 0

    # Step 3: Image Analysis
    try:
        df = run_image()
        analyzer_counts["Image"] = len(df) if df is not None and not df.empty else 0
    except Exception as e:
        print(f"\n  ⚠ Image Analyzer error: {e}")
        analyzer_counts["Image"] = 0

    # Step 4: Video Analysis
    try:
        df = run_video()
        analyzer_counts["Video"] = len(df) if df is not None and not df.empty else 0
    except Exception as e:
        print(f"\n  ⚠ Video Analyzer error: {e}")
        analyzer_counts["Video"] = 0

    # Step 5: Text Analysis
    try:
        df = run_text()
        analyzer_counts["Text"] = len(df) if df is not None and not df.empty else 0
    except Exception as e:
        print(f"\n  ⚠ Text Analyzer error: {e}")
        analyzer_counts["Text"] = 0

    # Step 6: Integration
    print("\n" + "#" * 60)
    print("#  INTEGRATION — Merging All Outputs")
    print("#" * 60)

    try:
        final_df = run_merge()
    except Exception as e:
        print(f"\n  ⚠ Integration error: {e}")
        final_df = None

    # Final summary
    print("\n" + "=" * 60)
    print("  PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    print("\n  Records processed per modality:")
    for modality, count in analyzer_counts.items():
        status = "✓" if count > 0 else "✗"
        print(f"    {status} {modality}: {count} records")

    if final_df is not None and not final_df.empty:
        print(f"\n  ✓ Final integrated report: {len(final_df)} records")
        print(f"    Saved to: output/final_integrated_report.csv")
    else:
        print(f"\n  ✗ No final report generated (no data processed)")

    print(f"\n  To view the dashboard, run:")
    print(f"    streamlit run integration/dashboard.py")
    print()


if __name__ == "__main__":
    main()
