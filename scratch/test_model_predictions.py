import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import sqlite3
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import MODEL_DIR

def clean_ocr_text(text: str) -> str:
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    # Remove strange characters and noise but keep words, numbers, and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,@#$:\-\/]', '', text)
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def predict(text, model, tokenizer):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, pred_idx = torch.max(probs, dim=-1)
        label = model.config.id2label[pred_idx.item()]
        return label, confidence.item(), probs[0].tolist()

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    
    conn = sqlite3.connect('documind.db')
    cur = conn.cursor()
    cur.execute('SELECT id, filename, category, confidence, extracted_text FROM documents WHERE id IN (35, 37)')
    rows = cur.fetchall()
    
    for row in rows:
        db_id, filename, old_label, old_conf, text = row
        print(f"\n=== DB ID: {db_id} | File: {filename} ===")
        print(f"Old Label: {old_label} ({old_conf:.2%})")
        print(f"Extracted Text (raw):\n{text[:300]}")
        
        # Predict on raw
        label_raw, conf_raw, probs_raw = predict(text, model, tokenizer)
        print(f"Predict Raw: {label_raw} ({conf_raw:.2%})")
        
        # Predict on cleaned
        cleaned = clean_ocr_text(text)
        label_clean, conf_clean, probs_clean = predict(cleaned, model, tokenizer)
        print(f"Predict Cleaned: {label_clean} ({conf_clean:.2%})")
        
        # Print all class probabilities
        classes = [model.config.id2label[i] for i in range(len(probs_clean))]
        print("Probabilities (Cleaned):")
        for c, p in zip(classes, probs_clean):
            print(f"  {c}: {p:.2%}")

if __name__ == "__main__":
    main()
