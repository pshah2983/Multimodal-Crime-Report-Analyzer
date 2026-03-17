"""
Incident Report Dashboard — Streamlit Application
===================================================
A simple interactive dashboard to display, filter, and query
the final integrated incident report dataset.

Run with:  streamlit run integration/dashboard.py
"""

import os
import sys
import pandas as pd
import streamlit as st

# Paths
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
FINAL_REPORT = os.path.join(OUTPUT_DIR, "final_integrated_report.csv")

# Individual modality CSVs
MODALITY_FILES = {
    "Audio Results": "audio_results.csv",
    "PDF Results": "pdf_results.csv",
    "Image Results": "image_results.csv",
    "Video Results": "video_results.csv",
    "Text Results": "text_results.csv",
}


def main():
    st.set_page_config(
        page_title="🚨 Multimodal Incident Report Analyzer",
        page_icon="🚨",
        layout="wide",
    )

    st.title("🚨 Multimodal Crime / Incident Report Analyzer")
    st.markdown("**AI-Powered Dashboard** — View and filter structured incident data extracted from 5 data modalities.")

    st.divider()

    # --- SECTION 1: Final Integrated Report ---
    st.header("📊 Final Integrated Report")

    if os.path.exists(FINAL_REPORT):
        df = pd.read_csv(FINAL_REPORT)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Incidents", len(df))
        col2.metric("High Severity", len(df[df["Severity"] == "High"]))
        col3.metric("Medium Severity", len(df[df["Severity"] == "Medium"]))
        col4.metric("Low Severity", len(df[df["Severity"] == "Low"]))

        # Filters
        st.subheader("🔍 Filter Incidents")
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )

        with filter_col2:
            search_term = st.text_input("Search across all columns", "")

        # Apply filters
        filtered = df[df["Severity"].isin(severity_filter)]
        if search_term:
            mask = filtered.apply(
                lambda row: search_term.lower() in " ".join(row.astype(str)).lower(),
                axis=1,
            )
            filtered = filtered[mask]

        st.dataframe(filtered, use_container_width=True, height=400)

        # Download button
        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered Report as CSV",
            data=csv_data,
            file_name="filtered_incident_report.csv",
            mime="text/csv",
        )
    else:
        st.warning(
            "⚠️ Final integrated report not found. "
            "Please run `python main.py` first to generate the report."
        )

    st.divider()

    # --- SECTION 2: Individual Modality Results ---
    st.header("📁 Individual Modality Results")

    for label, filename in MODALITY_FILES.items():
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            with st.expander(f"📄 {label} — {filename}"):
                mod_df = pd.read_csv(path)
                st.dataframe(mod_df, use_container_width=True)
                st.caption(f"{len(mod_df)} record(s)")
        else:
            with st.expander(f"⚠️ {label} — Not generated yet"):
                st.info(f"Run the {label.replace(' Results', '').lower()} analyzer to generate this file.")

    st.divider()

    # --- SECTION 3: Pipeline Info ---
    st.header("ℹ️ AI Pipeline Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    UNSTRUCTURED DATA SOURCES                       │
    │  🎤 Audio   📄 PDF   📷 Images   🎥 Video   📝 Text              │
    └──────┬──────────┬──────────┬──────────┬───────────┬───────────────┘
           │          │          │          │           │
           ▼          ▼          ▼          ▼           ▼
    ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
    │  Audio   ││   PDF    ││  Image   ││  Video   ││  Text    │
    │ Analyzer ││ Analyzer ││ Analyzer ││ Analyzer ││ Analyzer │
    │ (Gemini) ││(PyPDF2 + ││ (Gemini  ││(OpenCV + ││ (Gemini  │
    │          ││ Gemini)  ││  Vision) ││ Gemini)  ││   NLP)   │
    └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
         │           │           │           │           │
         ▼           ▼           ▼           ▼           ▼
    ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
    │  audio_  ││  pdf_    ││  image_  ││  video_  ││  text_   │
    │results   ││results   ││results   ││results   ││results   │
    │  .csv    ││  .csv    ││  .csv    ││  .csv    ││  .csv    │
    └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
         │           │           │           │           │
         └───────────┴───────────┴─────┬─────┴───────────┘
                                       ▼
                            ┌─────────────────────┐
                            │  INTEGRATION MODULE  │
                            │   merge.py           │
                            │  (pandas merge +     │
                            │   severity scoring)  │
                            └──────────┬──────────┘
                                       ▼
                            ┌─────────────────────┐
                            │  final_integrated_   │
                            │  report.csv          │
                            └──────────┬──────────┘
                                       ▼
                            ┌─────────────────────┐
                            │  STREAMLIT DASHBOARD │
                            │   dashboard.py       │
                            └─────────────────────┘
    ```
    """)

    st.markdown("---")
    st.caption("Built with ❤️ using Google Gemini AI & Streamlit | EDS 6344 AI for Engineers")


if __name__ == "__main__":
    main()
