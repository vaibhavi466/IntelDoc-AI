"""
inference.py — Document prediction pipeline for IntelDoc AI.

Performs OCR on a document image and classifies it using the
fine-tuned DistilBERT model. Requires a running Streamlit context
for @st.cache_resource to function.
"""
import streamlit as st  
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification  
import pytesseract  #Tesseract → extract text from images
import re
from PIL import Image  #Pillow → image handling
import os 

from src.config import MODEL_DIR

# Cross-platform Tesseract path configuration
tesseract_path = os.getenv("TESSERACT_CMD")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Cache at module level to avoid reloading on every call
@st.cache_resource
def load_model():
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()  # Set to evaluation mode
    return tokenizer, model

def apply_hybrid_rules(text: str, model_label: str, model_confidence: float) -> tuple[str, float]:
    text_lower = text.lower()
    
    # 1. INVOICE Heuristics (Robust to OCR typos)
    invoice_keywords = [
        'invoice', 'inv', 'bill to', 'purchase order', 'invoice number', 'total amount', 
        'total due', 'amount due', 'subtotal', 'unit price', 'quantity', 'qty', 'payment', 
        'balance due', 'tax', 'price', 'customer', 'customer id', 'client', 'remit to', 
        'due date', 'invoice date', 'grand total', 'description', 'charges', 'amount'
    ]
    has_invoice_word = any(w in text_lower for w in ['invoice', 'invoic', 'inv #', 'inv:', 'invno', 'inv_no', 'ewoice'])
    
    invoice_score = sum(1 for kw in invoice_keywords if kw in text_lower)
    if has_invoice_word:
        invoice_score += 2
    if 'total' in text_lower or 'tot ' in text_lower or 'tot:' in text_lower:
        invoice_score += 1
    if '#' in text_lower or 'no.' in text_lower or 'number' in text_lower:
        invoice_score += 1

    # 2. RESUME Heuristics
    resume_keywords = [
        'resume', 'curriculum vitae', 'education', 'work experience', 'experience', 
        'projects', 'skills', 'languages', 'hobbies', 'c.v.', 'c v', 'gpa', 'extracurricular'
    ]
    resume_score = sum(1 for kw in resume_keywords if kw in text_lower)
    if 'resume' in text_lower or 'curriculum vitae' in text_lower or 'c.v.' in text_lower:
        resume_score += 2
    if 'education' in text_lower and ('experience' in text_lower or 'skills' in text_lower):
        resume_score += 2

    # 3. EMAIL Heuristics
    email_indicators = ['subject:', 'from:', 'to:', 'sent:', 'cc:', 'bcc:']
    email_score = sum(1 for kw in email_indicators if kw in text_lower)
    has_email_headers = 'from:' in text_lower and ('to:' in text_lower or 'subject:' in text_lower or 'sent:' in text_lower)
    if has_email_headers:
        email_score += 3

    # 4. NEWS_ARTICLE Heuristics
    news_indicators = [
        'press release', 'associated press', 'reuters', 'hindustan times', 'times of', 
        'news service', 'reported by', 'published on', 'journal', 'gazette', 'editor', 
        'correspondent', 'reporternews', 'news desk', 'editorial'
    ]
    news_score = sum(1 for kw in news_indicators if kw in text_lower)

    # Heuristic Decisions
    if invoice_score >= 4 and model_label != 'invoice':
        return 'invoice', min(0.98, model_confidence + 0.35)
    
    if resume_score >= 4 and model_label != 'resume':
        return 'resume', min(0.98, model_confidence + 0.35)
        
    if email_score >= 4 and model_label != 'email':
        return 'email', min(0.98, model_confidence + 0.35)

    # Boosting confidence for matching cases
    boosted_confidence = model_confidence
    if model_label == 'invoice' and invoice_score >= 1:
        boosted_confidence = min(0.99, model_confidence + 0.15)
    elif model_label == 'news_article' and news_score >= 1:
        boosted_confidence = min(0.99, model_confidence + 0.20)
    elif model_label == 'resume' and resume_score >= 2:
        boosted_confidence = min(0.99, model_confidence + 0.15)
    elif model_label == 'email' and email_score >= 1:
        boosted_confidence = min(0.99, model_confidence + 0.15)

    return model_label, boosted_confidence

def predict_document(image_path: str)->tuple[str, float, str]:
    # Step 1: Load cached model
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        return "Model not found", 0.0, ""
    
    # step:1b. OCR: Get text from image
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        return f"Error reading image: {e}", 0.0, ""

    if not text.strip():
        return "No text found in document", 0.0, ""


    # Step: 2. Prepare Text for model input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

    # Step: 3. Run Inference
    with torch.no_grad():
        outputs = model(**inputs)  
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1) 
        # Get the highest probability
        confidence, predicted_class_idx = torch.max(probs, dim=-1)
        # Get the label name (e.g., "invoice")
        label = model.config.id2label[predicted_class_idx.item()]
    
    # Apply hybrid heuristic rules to refine prediction and boost confidence
    final_label, final_confidence = apply_hybrid_rules(text, label, confidence.item())
    
    return final_label, final_confidence, text

if __name__ == "__main__":
    # Test with a dummy path
    print("Inference script ready. Run via App.")