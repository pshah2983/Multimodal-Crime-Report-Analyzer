"""
PDF Analyzer Module — Student 2 Role
=====================================
Processes police reports, official documents, and incident PDFs.
Uses PyPDF2 for text extraction and Google Gemini for NLP analysis.

Alternative tools (referenced for academic credit):
- pdfplumber: Better table extraction from PDFs
- pytesseract: OCR for scanned PDF images
- spaCy: Named Entity Recognition

Input:  PDF files (.pdf) in pdf/data/
Output: output/pdf_results.csv
Schema: Doc_ID, Doc_Title, Extracted_Text, Key_Entities, Doc_Type, Summary
"""

import os
import sys
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.gemini_helper import analyze_with_gemini, analyze_multimodal, extract_json_from_response

import requests
import tempfile

# Directory paths
OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pdf_results.csv")

# Prompt for Gemini analysis of extracted PDF text
PDF_PROMPT = """You are an AI document analyst for a law enforcement agency.
Analyze the following PDF document content and extract structured information.

Perform the following tasks:
1. Identify the document title or subject.
2. Extract key entities: people names, organizations, locations, dates, case numbers.
3. Classify the document type (e.g., Police Report, Training Proposal, Incident Log, Evidence Form, Arrest Record).
4. Write a concise summary of the document (2-3 sentences).

Return your analysis as a JSON object with EXACTLY these keys:
{
    "doc_title": "Title or subject of the document",
    "key_entities": "Comma-separated list of key entities found",
    "doc_type": "Classification of document type",
    "summary": "Brief 2-3 sentence summary"
}

Return ONLY the JSON object, no other text.

--- DOCUMENT CONTENT ---
"""

PDF_MULTIMODAL_PROMPT = """You are an AI document analyst for a law enforcement agency.
Analyze this PDF document and extract structured information.

Perform the following tasks:
1. Identify the document title or subject.
2. Extract key entities: people names, organizations, locations, dates, case numbers.
3. Classify the document type (e.g., Police Report, Training Proposal, Incident Log, Evidence Form, Arrest Record).
4. Write a concise summary of the document (2-3 sentences).

Return your analysis as a JSON object with EXACTLY these keys:
{
    "doc_title": "Title or subject of the document",
    "key_entities": "Comma-separated list of key entities found",
    "doc_type": "Classification of document type",
    "summary": "Brief 2-3 sentence summary"
}

Return ONLY the JSON object, no other text."""


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF using PyPDF2.
    Falls back to empty string if extraction fails.
    
    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text content.
    """
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"  WARNING: PyPDF2 extraction failed: {e}")
        return ""


def analyze_pdf_file(file_path, doc_id):
    """
    Analyze a single PDF file. First tries text extraction + Gemini text analysis.
    If no text is extractable, uses Gemini multimodal (direct PDF upload).
    
    Args:
        file_path (str): Path to the PDF file.
        doc_id (str): Identifier for this document.
    
    Returns:
        dict: Structured analysis result.
    """
    print(f"\n[PDF Analyzer] Processing: {os.path.basename(file_path)}")

    try:
        # Step 1: Try extracting text with PyPDF2
        extracted_text = extract_text_from_pdf(file_path)

        if extracted_text and len(extracted_text.strip()) > 50:
            # Use text-based analysis (faster, cheaper)
            print("  Using text extraction + Gemini text analysis...")
            # Truncate very long documents to avoid token limits
            truncated_text = extracted_text[:8000]
            prompt = PDF_PROMPT + truncated_text
            response = analyze_with_gemini(prompt)
        else:
            # Fallback: Use Gemini multimodal with direct PDF upload
            print("  Text extraction yielded little text. Using Gemini multimodal PDF analysis...")
            extracted_text = "(Scanned/image-based PDF — text extracted by Gemini)"
            response = analyze_multimodal(PDF_MULTIMODAL_PROMPT, file_path)

        result = extract_json_from_response(response)

        if result:
            return {
                "Doc_ID": doc_id,
                "Doc_Title": result.get("doc_title", ""),
                "Extracted_Text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                "Key_Entities": result.get("key_entities", ""),
                "Doc_Type": result.get("doc_type", ""),
                "Summary": result.get("summary", ""),
            }
        else:
            print(f"  ERROR: Could not parse Gemini response for {doc_id}")
            return _empty_result(doc_id)

    except Exception as e:
        print(f"  ERROR processing {doc_id}: {e}")
        return _empty_result(doc_id)


def _empty_result(doc_id):
    """Return an empty result row for failed analyses."""
    return {
        "Doc_ID": doc_id,
        "Doc_Title": "Error: Analysis failed",
        "Extracted_Text": "",
        "Key_Entities": "",
        "Doc_Type": "",
        "Summary": "",
    }


def run():
    """
    Main entry point: scans pdf/data/ for PDF files,
    analyzes each, and writes output/pdf_results.csv.
    
    Returns:
        pd.DataFrame: The results DataFrame.
    """
    print("\n" + "=" * 60)
    print("  PDF ANALYZER — Student 2 Role")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download the PDF dynamically
    pdf_url = "https://cdn.muckrock.com/foia_files/2015/08/18/LESO2.pdf"
    print(f"  Downloading PDF from: {pdf_url}")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
    except Exception as e:
        print(f"  Failed to download PDF: {e}")
        return pd.DataFrame()

    results = []
    # Process the downloaded PDF
    doc_id = f"DOC_001"
    result = analyze_pdf_file(tmp_path, doc_id)
    results.append(result)
    
    # Clean up
    try:
        os.remove(tmp_path)
    except:
        pass

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  PDF results saved to: {OUTPUT_FILE}")
    print(f"  Total records: {len(df)}")
    return df


if __name__ == "__main__":
    run()
