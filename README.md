Git Repo Link : https://github.com/pshah2983/Multimodal-Crime-Report-Analyzer

# Multimodal Crime & Incident Report Analyzer

An AI-powered modular processing pipeline that analyzes multimodal emergency data streams (audio, document, image, video, and text). It uses various Google Gemini models and computer vision tools to extract critical incident information and merge it into a single, unified severity report.

## 🗂 Project Structure

```text
multimodal_final_submission/
├── audio/            # Audio Analyzer (Student 1) - Processes 911 calls
├── pdf/              # PDF Analyzer (Student 2) - Processes Official Police Docs
├── images/           # Image Analyzer (Student 3) - Processes Crime Scene Photos
├── video/            # Video Analyzer (Student 4) - Processes Surveillance Video
├── text/             # Text Analyzer (Student 5) - Processes Crime Report Texts
├── integration/      # Integration Module - Merges all multimodal data
├── output/           # Final CSV outputs from all modules
└── main.py           # Master execution script
```

## 🚀 How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
pip install kagglehub
```
Make sure you have a `.env` file in the root directory with your API key:
`GEMINI_API_KEY=your_key_here`

### Run the Full Pipeline
```bash
python main.py
```
This single command runs all 5 analyzers in sequence, followed by the integration script to generate `final_integrated_report.csv`.

### Run Individual Modules
```bash
python main.py --audio     # Only Audio Analyzer
python main.py --pdf       # Only PDF Analyzer
python main.py --image     # Only Image Analyzer
python main.py --video     # Only Video Analyzer
python main.py --text      # Only Text Analyzer
python main.py --merge     # Only Integration/Merge step
```

---

## 📥 Datasets

The system's data is verified to use the following exact sources as requested:

### 🎤 1. Audio — Emergency Call Recordings
- **Dataset Source:** [Kaggle - 911 Calls Wav2Vec2](https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2) / [911 Recordings: The First 6 Seconds Dataset](https://www.kaggle.com/datasets/louisteitelbaum/911-recordings-first-6-seconds)
- **Usage:** Dynamically fetched at runtime directly within `audio/audio_analyzer.py` using `kagglehub`.
- **How to Fetch:** You do NOT need to download anything manually. The codebase contains the following logic to automatically fetch the dataset on the fly:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("louisteitelbaum/911-recordings-first-6-seconds")
  ```

### 📄 2. Document — Police Department Documents
- **Dataset Source:** [MuckRock (FOIA-released official police documents)](https://www.muckrock.com/foi/arkansas-114/arkansas-police-departments-1033-training-plan-proposals-20493/#file-52365)
- **Usage:** The script `pdf/pdf_analyzer.py` directly issues an HTTP request to download the Document `LESO2.pdf` straight from the source URL at runtime. No manual download is necessary.

### 📷 3. Images — Fire / Incident Scene Detection
- **Dataset Source:** [Roboflow Fire Detection](https://universe.roboflow.com/search?q=fire)
- **Usage:** Placed centrally inside `input_datasets/images/`.

### 🎥 4. Video — CCTV Surveillance Footage
- **Dataset Source:** [CAVIAR Dataset (homepages.inf.ed.ac.uk)](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)
- **Usage:** Placed centrally inside `input_datasets/video/`.

### 📝 5. Text — Crime Report Text Data
- **Dataset Source:** [Kaggle - CrimeReport](https://www.kaggle.com/datasets/cameliasiadat/crimereport)
- **Usage:** Dynamically fetched at runtime directly within `text/text_analyzer.py` using `kagglehub`.
- **How to Fetch:** You do NOT need to download anything manually. The exact dataset is fetched automatically during script execution via:
  ```python
  import kagglehub
  data_path = kagglehub.dataset_download("cameliasiadat/crimereport")
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
| `Sentiment` | Speaker sentiment |
| `Urgency_Score` | Urgency score from 0.0 to 1.0 |

**PDF** (`pdf_results.csv`):
| Column | Description |
|--------|-------------|
| `Doc_ID` | Unique identifier |
| `Doc_Title` | Document title or subject |
| `Extracted_Text` | Extracted text snipped to 500 chars |
| `Key_Entities` | People, organizations, locations, dates |
| `Doc_Type` | Document classification |
| `Summary` | Concise 2-3 sentence summary |

**Image** (`image_results.csv`):
| Column | Description |
|--------|-------------|
| `Image_ID` | Unique identifier |
| `Scene_Type` | Scene classification (Fire Scene, etc.) |
| `Objects_Detected` | List of objects found |
| `Bounding_Boxes` | Approximate regions of detected objects |
| `Confidence` | Confidence score from 0.0 to 1.0 |

**Video** (`video_results.csv`):
| Column | Description |
|--------|-------------|
| `Clip_ID` | Unique identifier |
| `Timestamp` | Time in the video when event occurs |
| `Frame_ID` | Frame identifier |
| `Event_Detected` | Description of detected event/activity |
| `Persons_Count` | Number of persons visible |
| `Confidence` | Confidence score from 0.0 to 1.0 |

**Text** (`text_results.csv`):
| Column | Description |
|--------|-------------|
| `Text_ID` | Unique identifier |
| `Crime_Type` | Type of crime identified |
| `Location_Entity` | Locations extracted via NER |
| `Sentiment` | Overall sentiment |
| `Topic` | Topic classification |
| `Severity_Label` | Severity rating |

### Final Integrated Output (`final_integrated_report.csv`)

| Column | Source |
|--------|--------|
| `Incident_ID` | Auto-generated (INC_001, ...) |
| `Severity` | Calculated from all modality signals (High / Medium / Low) |
| *(Additional)*| Merged fields from Audio, PDF, Image, Video, Text |

---

## 🧠 Team Roles & Technology Mapping

| Role | Student | Data Type | Tools |
|------|---------|-----------|-------|
| **Audio Analyst** | Student 1 | 911 calls (.wav) | Gemini Audio API |
| **PDF Analyst** | Student 2 | Police reports (.pdf)| PyPDF2 + Gemini NLP |
| **Image Analyst** | Student 3 | Scene photos (.jpg)| Gemini Vision API |
| **Video Analyst** | Student 4 | CCTV (.mpg) | OpenCV + Gemini Vision |
| **Text Analyst** | Student 5 | Crime tweets (.txt) | Gemini NLP |
| **Integration** | All | Multi-CSV | pandas, Gemini logic |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `GEMINI_API_KEY not found` | Add `GEMINI_API_KEY=your_key_here` to `.env` |
| `Missing Data Error` | The `input_datasets/` folder must be present at the root for Images and Videos. Audio, PDF, and Text are completely dynamic and no manual download is necessary. |
| `API quota exceeded` | Wait a few minutes (Gemini Free Tier Rate Limits) |
