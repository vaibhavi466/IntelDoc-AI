# DocumindAI 📄 — Intelligent Document Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-red)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-yellow)

## Overview
An end-to-end ML system that classifies scanned documents (invoices, 
emails, resumes, etc.) using a fine-tuned DistilBERT model, with OCR 
text extraction, NER-based entity extraction, and AI summarization.

## Architecture
- **OCR**: Tesseract → raw text from document images
- **Classifier**: DistilBERT fine-tuned on RVL-CDIP categories
- **NER**: SpaCy en_core_web_sm → people, organizations, dates
- **Summarization**: DistilBART CNN → abstractive summaries
- **UI**: Streamlit with SQLite document history

## Setup & Run
```bash
pip install -r requirements.txt
python src/create_dataset.py   # Step 1: Build dataset
python src/train_model.py      # Step 2: Fine-tune model
python src/evaluate.py         # Step 3: Evaluate
streamlit run app.py           # Step 4: Launch app
```

## Results
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| DistilBERT (fine-tuned) | 91.2% | 0.89 |