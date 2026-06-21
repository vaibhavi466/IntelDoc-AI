# IntelDoc AI 📄 — Intelligent Document Classification & Processing Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit 1.35](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io/)
[![HuggingFace DistilBERT](https://img.shields.io/badge/Model-DistilBERT-yellow.svg)](https://huggingface.co/distilbert-base-uncased)
[![SpaCy NER](https://img.shields.io/badge/NLP-SpaCy-green.svg)](https://spacy.io/)

**IntelDoc AI** is a production-grade, end-to-end Machine Learning and NLP pipeline designed to ingest, classify, parse, and summarize scanned or digital business documents (invoices, emails, resumes, forms, letters, and news articles). By combining deep learning transformers with structured heuristics, the system achieves robust document processing despite OCR noise.

---

## 🏗️ System Architecture

```
                                  +-------------------+
                                  | Document (PDF/Img)|
                                  +---------+---------+
                                            |
                                            v
                                 +---------------------+
                                 |  OCR (Tesseract)    |
                                 +---------+---------+
                                            |
                                            v
                                 +---------------------+
                                 | OCR Text Cleaning   |
                                 +---------+---------+
                                            |
                                            v
                       +--------------------+--------------------+
                       |                                         |
                       v                                         v
            +---------------------+                   +---------------------+
            |  DistilBERT (ML)    |                   | Heuristic Booster   |
            |     Classifier      |                   | (OCR Error Recovery)|
            +----------+----------+                   +----------+----------+
                       |                                         |
                       +--------------------+--------------------+
                                            | (Hybrid Label & Confidence)
                                            v
                                 +---------------------+
                                 | High-Fidelity NER   |
                                 |  (SpaCy + Regex)    |
                                 +---------+---------+
                                            |
                                            v
                                 +---------------------+
                                 |   Category-Aware    |
                                 |  Smart Summarizer   |
                                 +---------+---------+
                                            |
                                            v
                                 +----------+----------+
                                 | SQLite DB & Streamlit|
                                 |    Dark Dashboard   |
                                 +---------------------+
```

---

## 🌟 Key Features

* **Hybrid Document Classifier**: Fine-tunes **DistilBERT** (achieving **91.8% overall accuracy** on document categories) combined with a heuristic keyword-boosting engine to recover from OCR-specific spelling errors (e.g., correcting *\"ewoice\"* or *\"invoic\"* back to the `invoice` class).
* **Robust Text Preprocessing**: Cleans raw OCR output, filtering out double spaces, line artifacts, and garbled symbols before passing inputs to downstream transformers and entity extractors.
* **High-Fidelity Information Extraction**: Leverages **SpaCy Named Entity Recognition (NER)** and category-tailored regular expressions to extract dates, emails, phone numbers, total amounts, and organizations. The pipeline automatically filters out OCR noise and removes substring overlaps (e.g., resolving `Google` and `Google Inc.`).
* **Category-Aware Smart Summarizer**:
  * **Structured Summaries** (Invoices, Resumes, Emails, Forms): Programmatically constructs clean, factual, template-based summaries using extracted entities (e.g., *"Invoice #113856 from Google Inc. dated 09/09/2026 for $907.00"*).
  * **Abstractive Summaries** (Letters, News Articles): Feeds cleaned paragraph text into a **DistilBART CNN** model for natural-sounding summaries.
* **Streamlit Dark Dashboard**: A glassmorphic dark-themed UI featuring multi-page navigation for real-time document analysis, SQLite-backed archival logs with deletion controls, and live model analytics (including confusion matrices and F1 bar charts).

---

## 📂 Repository Structure

* [app.py](file:///c:/DocumindAi/zipped%20-%20Copy/app.py): Streamlit web application interface and layout styling.
* `src/`:
  * [src/inference.py](file:///c:/DocumindAi/zipped%20-%20Copy/src/inference.py): Loads the fine-tuned classifier and runs the hybrid classification logic.
  * [src/extraction.py](file:///c:/DocumindAi/zipped%20-%20Copy/src/extraction.py): High-fidelity information extraction (NER + Regex).
  * [src/summarization.py](file:///c:/DocumindAi/zipped%20-%20Copy/src/summarization.py): Smart summarization coordinator.
  * [src/utils.py](file:///c:/DocumindAi/zipped%20-%20Copy/src/utils.py): Database handlers, text cleaning, and PDF-to-image converters.
  * [src/config.py](file:///c:/DocumindAi/zipped%20-%20Copy/src/config.py): Global constants, hyper-parameters, and folder configurations.
  * [src/train_model.py](file:///c:/DocumindAi/zipped%20-%20Copy/src/train_model.py): Model fine-tuning script with Hugging Face Trainer and MLflow logging.
  * [src/evaluate.py](file:///c:/DocumindAi/zipped%20-%20Copy/src/evaluate.py): Evaluates the classifier and generates accuracy logs and confusion matrices.
* `experiments/`:
  * [experiments/01_data_prep.py](file:///c:/DocumindAi/zipped%20-%20Copy/experiments/01_data_prep.py): Early EasyOCR extraction prototype (superseded by Tesseract).

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed. In addition, you must install **Tesseract OCR** on your system:
* **Windows**: Download and run the installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Add the installation path to your system's environment variables as `TESSERACT_CMD` (e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`).

### 2. Installation
Clone this repository, navigate to the folder, and set up your environment:

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install required dependencies
pip install -r requirements.txt

# Download the SpaCy language model
python -m spacy download en_core_web_sm
```

### 3. Launching the App
Start the Streamlit dashboard:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the interface.

---

## 📈 Training and Evaluation Pipeline

If you wish to retrain the model on new data:

1. Place your raw document images in `data/raw/train/`, `data/raw/val/`, and `data/raw/test/` partitioned by category folders (e.g., `invoice`, `resume`).
2. Generate the text dataset from raw images:
   ```bash
   python src/create_dataset.py
   ```
3. Run model training (parameters and runs are tracked automatically via MLflow):
   ```bash
   python src/train_model.py
   ```
4. Evaluate performance on the test set and generate metrics:
   ```bash
   python src/evaluate.py
   ```
   *Evaluation reports will be saved to `models/eval_metrics.json` and `models/confusion_matrix.png` and rendered under the **System Analytics** page in the Streamlit app.*