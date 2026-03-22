# 🚨 Multimodal Crime / Incident Report Analyzer

An **AI-powered pipeline** that processes unstructured data from five different modalities — **Audio, PDFs, Images, Video, and Text** — and converts them into a single **structured incident report dataset**. Built using **Google Gemini AI** as the primary analysis engine.

> **Course:** EDS 6344 — AI for Engineers  
> **Assignment:** Group Assignment — Multimodal Incident Analyzer  
> **Team Size:** 5 Students

---

## 📐 AI Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       UNSTRUCTURED DATA SOURCES                        │
│   🎤 Audio (.mp3/.wav)   📄 PDF (.pdf)   📷 Images (.jpg/.png)        │
│   🎥 Video (.mp4/.mpg)   📝 Text (.txt/.csv)                          │
└────┬──────────┬──────────────┬──────────────┬──────────────┬───────────┘
     │          │              │              │              │
     ▼          ▼              ▼              ▼              ▼
┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  AUDIO  │ │   PDF   │ │  IMAGE   │ │  VIDEO   │ │  TEXT    │
│ANALYZER │ │ANALYZER │ │ ANALYZER │ │ ANALYZER │ │ ANALYZER │
│         │ │         │ │          │ │          │ │          │
│ Gemini  │ │ PyPDF2  │ │  Gemini  │ │ OpenCV + │ │  Gemini  │
│ Audio   │ │+Gemini  │ │  Vision  │ │  Gemini  │ │   NLP    │
│ API     │ │  NLP    │ │          │ │  Vision  │ │          │
└────┬────┘ └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │           │           │            │             │
     ▼           ▼           ▼            ▼             ▼
┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ audio_  │ │  pdf_   │ │ image_   │ │ video_   │ │ text_    │
│results  │ │results  │ │ results  │ │ results  │ │ results  │
│  .csv   │ │  .csv   │ │  .csv    │ │  .csv    │ │  .csv    │
└────┬────┘ └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │           │           │            │             │
     └───────────┴───────────┴─────┬──────┴─────────────┘
                                   ▼
                      ┌────────────────────────┐
                      │   INTEGRATION MODULE   │
                      │      merge.py          │
                      │                        │
                      │  • Assign Incident_IDs │
                      │  • Merge all 5 CSVs    │
                      │  • Severity scoring    │
                      └───────────┬────────────┘
                                  ▼
                      ┌────────────────────────┐
                      │  final_integrated_     │
                      │  report.csv            │
                      └───────────┬────────────┘
                                  ▼
                      ┌────────────────────────┐
                      │  STREAMLIT DASHBOARD   │
                      │  dashboard.py          │
                      │                        │
                      │  • Filter & search     │
                      │  • Severity metrics    │
                      │  • CSV download        │
                      └────────────────────────┘
```

---

## 📁 Project Structure

```
Multimodal Crime Report Analyzer/
│
├── README.md                   # This file — project documentation
├── requirements.txt            # Python dependencies
├── .env                        # API key configuration (not committed to Git)
├── .gitignore                  # Git ignore rules
├── main.py                     # 🚀 Main orchestrator — runs the full pipeline
│
├── audio/                      # Student 1 — Audio Analyst
│   ├── audio_analyzer.py       # Transcribes audio + extracts event/location/sentiment
│   └── data/                   # Place .mp3/.wav audio files here
│
├── pdf/                        # Student 2 — PDF Analyst
│   ├── pdf_analyzer.py         # Extracts text from PDFs + classifies document type
│   └── data/                   # Place .pdf files here
│
├── images/                     # Student 3 — Image Analyst
│   ├── image_analyzer.py       # Detects objects, classifies scene, performs OCR
│   └── data/                   # Place .jpg/.png images here
│
├── video/                      # Student 4 — Video Analyst
│   ├── video_analyzer.py       # Extracts frames + detects events/activities
│   └── data/                   # Place .mp4/.mpg video clips here
│
├── text/                       # Student 5 — Text Analyst
│   ├── text_analyzer.py        # NER, sentiment, topic classification on text/CSV
│   └── data/                   # Place .txt/.csv text data here
│
├── integration/                # Final Integration (Team Effort)
│   ├── merge.py                # Merges all 5 CSVs into unified dataset
│   └── dashboard.py            # Streamlit dashboard for visualization
│
├── utils/                      # Shared Utilities
│   └── gemini_helper.py        # Google Gemini API interface
│
└── output/                     # Generated output CSVs
    ├── audio_results.csv
    ├── pdf_results.csv
    ├── image_results.csv
    ├── video_results.csv
    ├── text_results.csv
    └── final_integrated_report.csv
```

---

## 🛠️ Setup Instructions

### Prerequisites

- **Python 3.9+** installed on your system
- A **Google Gemini API key** (get one from [Google AI Studio](https://aistudio.google.com/apikey))

### Step 1: Clone / Download the Repository

```bash
git clone <repository-url>
cd "Multimodal Crime Report Analyzer"
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Your API Key

Edit the `.env` file and replace the placeholder with your actual Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

### Step 5: Add Sample Data

Place your data files into the corresponding `data/` folders:

| Folder         | File Types       | What to Put Here                            |
|----------------|------------------|---------------------------------------------|
| `audio/data/`  | `.mp3`, `.wav`   | Emergency call recordings, voice statements |
| `pdf/data/`    | `.pdf`           | Police reports, official incident documents |
| `images/data/` | `.jpg`, `.png`   | Crime scene photos, accident images         |
| `video/data/`  | `.mp4`, `.mpg`   | CCTV clips, surveillance footage            |
| `text/data/`   | `.txt`, `.csv`   | Crime text reports, social media posts      |

> See the **Recommended Datasets** section below for where to download sample data.

### Step 6: Run the Full Pipeline

```bash
python main.py
```

This will:
1. Process all files in each `data/` folder
2. Generate 5 individual result CSVs in `output/`
3. Merge everything into `output/final_integrated_report.csv`

### Step 7: Launch the Dashboard

```bash
streamlit run integration/dashboard.py
```

Open `http://localhost:8501` in your browser to view and filter the results.

---

## ▶️ Usage

### Run Full Pipeline
```bash
python main.py
```

### Run Individual Analyzers
```bash
python main.py --audio     # Only Audio Analyzer
python main.py --pdf       # Only PDF Analyzer
python main.py --image     # Only Image Analyzer
python main.py --video     # Only Video Analyzer
python main.py --text      # Only Text Analyzer
python main.py --merge     # Only Integration/Merge step
```

### Run a Module Directly
```bash
python -m audio.audio_analyzer
python -m pdf.pdf_analyzer
python -m images.image_analyzer
python -m video.video_analyzer
python -m text.text_analyzer
```

---

## 📊 Output Schema

### Individual Modality Outputs

**Audio** (`audio_results.csv`):
| Column | Description |
|--------|-------------|
| `Call_ID` | Unique identifier (C001, C002, ...) |
| `Transcript` | Full transcription of the audio |
| `Extracted_Event` | Type of incident detected |
| `Location` | Location mentioned in the audio |
| `Sentiment` | Speaker sentiment (Calm / Concerned / Distressed / Panicked) |
| `Urgency_Score` | Urgency score from 0.0 to 1.0 |

**PDF** (`pdf_results.csv`):
| Column | Description |
|--------|-------------|
| `Doc_ID` | Unique identifier (DOC_001, DOC_002, ...) |
| `Doc_Title` | Document title or subject |
| `Extracted_Text` | First 500 characters of extracted text |
| `Key_Entities` | People, organizations, locations, dates |
| `Doc_Type` | Document classification (Police Report, Training Proposal, etc.) |
| `Summary` | Concise 2-3 sentence summary |

**Image** (`image_results.csv`):
| Column | Description |
|--------|-------------|
| `Image_ID` | Unique identifier (IMG_001, IMG_002, ...) |
| `Scene_Type` | Scene classification (Fire Scene, Traffic Accident, etc.) |
| `Objects_Detected` | List of objects found (fire, smoke, vehicles, etc.) |
| `Bounding_Boxes` | Approximate regions of detected objects |
| `Confidence` | Confidence score from 0.0 to 1.0 |

**Video** (`video_results.csv`):
| Column | Description |
|--------|-------------|
| `Clip_ID` | Unique identifier (CLIP_001, CLIP_002, ...) |
| `Timestamp` | Time in the video when event occurs |
| `Frame_ID` | Frame identifier (FRM_001, ...) |
| `Event_Detected` | Description of detected event/activity |
| `Persons_Count` | Number of persons visible |
| `Confidence` | Confidence score from 0.0 to 1.0 |

**Text** (`text_results.csv`):
| Column | Description |
|--------|-------------|
| `Text_ID` | Unique identifier (TXT_001, TXT_002, ...) |
| `Crime_Type` | Type of crime identified |
| `Location_Entity` | Locations extracted via NER |
| `Sentiment` | Overall sentiment (Positive / Negative / Neutral) |
| `Topic` | Topic classification (Theft, Violence, Fire, etc.) |
| `Severity_Label` | Severity rating (Low / Medium / High / Critical) |

### Final Integrated Output

**`final_integrated_report.csv`**:
| Column | Source |
|--------|--------|
| `Incident_ID` | Auto-generated (INC_001, ...) |
| `Audio_Event` | From Audio Analyzer |
| `Audio_Sentiment` | From Audio Analyzer |
| `Urgency_Score` | From Audio Analyzer |
| `PDF_Doc_Type` | From PDF Analyzer |
| `PDF_Summary` | From PDF Analyzer |
| `Image_Objects` | From Image Analyzer |
| `Image_Scene` | From Image Analyzer |
| `Video_Event` | From Video Analyzer |
| `Text_Crime_Type` | From Text Analyzer |
| `Text_Topic` | From Text Analyzer |
| `Severity` | Calculated from all modality signals (High / Medium / Low) |

---

## 📥 Recommended Datasets

Below are the recommended datasets for each modality. All are free to access.

### 🎤 Audio — Emergency Call Recordings
- **Option 1:** Search YouTube for "911 emergency call recordings" and download using `yt-dlp` as mp3/wav
- **Option 2:** Use [Mozilla Common Voice](https://commonvoice.mozilla.org/) for general speech samples
- **Tip:** Even 2-3 short audio clips (30-60 seconds each) are sufficient for a prototype

### 📄 PDF — Police Department Documents
- **Dataset:** Arkansas Police Department 1033 Training Plan Proposals
- **Source:** MuckRock (FOIA-released official police documents)
- **Link:** [muckrock.com Arkansas Police PDF](https://www.muckrock.com)
- **Access:** Open the link, scroll to the file section, download the PDF directly. No account required.
- **Tip:** This is a text-based PDF, so PyPDF2 will extract it cleanly.

### 📷 Images — Fire / Incident Scene Detection
- **Dataset:** Roboflow Fire Detection (pre-labeled fire and smoke images)
- **Link:** [universe.roboflow.com Fire Detection](https://universe.roboflow.com/search?q=fire+detection)
- **Access:** Pick a dataset with 1000+ images, click Download, choose YOLOv8 format. Free with Roboflow account.
- **Tip:** Download just 5-10 images for testing.

### 🎥 Video — CCTV Surveillance Footage
- **Dataset:** CAVIAR CCTV Dataset (simulated indoor surveillance)
- **Link:** [homepages.inf.ed.ac.uk/rbf/CAVIARDATA1](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)
- **Access:** Browse scenario folders, download `.mpg` clips directly. No account needed.
- **Recommended clips:** Start with `Browse` or `OneStopEnter` folders, then `Fight` or `Collapse` scenarios.
- **Tip:** Download only 3-5 short clips.

### 📝 Text — Crime Report Text Data
- **Dataset:** CrimeReport — Kaggle dataset with real crime text reports
- **Link:** [kaggle.com/datasets/cameliasiadat/crimereport](https://www.kaggle.com/datasets/cameliasiadat/crimereport)
- **Access:** Sign into Kaggle, open the link, click "Download". Place the CSV in `text/data/`.
- **Tip:** The dataset is already in CSV format — the text analyzer will auto-detect it.

---

## 🧠 Team Roles & Technology Mapping

| Role | Student | Data Type | Primary Tool | Alternative Tools |
|------|---------|-----------|-------------|-------------------|
| **Audio Analyst** | Student 1 | Emergency calls (.mp3, .wav) | Google Gemini Audio API | OpenAI Whisper, spaCy |
| **PDF Analyst** | Student 2 | Police reports (.pdf) | PyPDF2 + Gemini NLP | pdfplumber, pytesseract, spaCy |
| **Image Analyst** | Student 3 | Scene photos (.jpg, .png) | Gemini Vision API | YOLOv8, OpenCV, pytesseract |
| **Video Analyst** | Student 4 | CCTV footage (.mp4, .mpg) | OpenCV + Gemini Vision | YOLOv8, PyTorch, moviepy |
| **Text Analyst** | Student 5 | Crime reports (.txt, .csv) | Gemini NLP | spaCy, HuggingFace transformers, NLTK |

---

## 🧪 Tools & Libraries Used

| Library | Purpose | Install |
|---------|---------|---------|
| `google-generativeai` | Google Gemini API for multimodal AI analysis | `pip install google-generativeai` |
| `pandas` | Data manipulation and CSV generation | `pip install pandas` |
| `python-dotenv` | Load API keys from `.env` file | `pip install python-dotenv` |
| `Pillow` | Image validation and preprocessing | `pip install Pillow` |
| `PyPDF2` | PDF text extraction | `pip install PyPDF2` |
| `opencv-python` | Video frame extraction and processing | `pip install opencv-python` |
| `streamlit` | Interactive web dashboard | `pip install streamlit` |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `GEMINI_API_KEY not found` | Make sure your `.env` file contains `GEMINI_API_KEY=your_key_here` |
| `No audio/PDF/image files found` | Place files in the correct `<modality>/data/` folder |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `API quota exceeded` | Gemini free tier has rate limits. Wait a few minutes or use fewer files |
| Video analysis is slow | Reduce `FRAMES_PER_VIDEO` in `video/video_analyzer.py` or use shorter clips |
| Dashboard won't load | Make sure you run from the project root: `streamlit run integration/dashboard.py` |

---

## 📝 License

This project is for academic purposes as part of the EDS 6344 — AI for Engineers course.

---

## 👥 Contributors

| Name | Role | Contribution |
|------|------|-------------|
| Smit Patel | Audio Analyst | Audio transcription and event extraction |
| Nisarg Shah | PDF Analyst | PDF parsing and document classification |
| Somil Doshi | Image Analyst | Scene classification and object detection |
| Parva Shah | Video Analyst | Frame extraction and event detection |
| Md Moshiur Rahman | Text Analyst | NER, sentiment analysis, and topic classification |
| All | Integration | Data merging, severity scoring, and dashboard |
