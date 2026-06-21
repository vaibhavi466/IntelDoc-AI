import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import re

def apply_hybrid_rules(text: str, model_label: str, model_confidence: float) -> tuple[str, float]:
    text_lower = text.lower()
    
    # 1. INVOICE Heuristics (Robust to OCR typos)
    # Check for keywords
    invoice_keywords = [
        'invoice', 'inv', 'bill to', 'purchase order', 'invoice number', 'total amount', 
        'total due', 'amount due', 'subtotal', 'unit price', 'quantity', 'qty', 'payment', 
        'balance due', 'tax', 'price', 'customer', 'customer id', 'client', 'remit to', 
        'due date', 'invoice date', 'grand total', 'description', 'charges', 'amount'
    ]
    # Check for OCR typos like 'ewoice' or 'invoic'
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
    # Check for email headers
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

    print(f"Scores -> Invoice: {invoice_score}, Resume: {resume_score}, Email: {email_score}, News: {news_score}")

    # Heuristic Decisions
    # Threshold rules for overrides
    # If it has strong invoice features and the model predicted letter/form, override it
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

def main():
    conn = sqlite3.connect('documind.db')
    cur = conn.cursor()
    cur.execute('SELECT id, filename, category, confidence, extracted_text FROM documents WHERE id IN (35, 37)')
    rows = cur.fetchall()
    
    for row in rows:
        db_id, filename, old_label, old_conf, text = row
        print(f"\n=== DB ID: {db_id} | File: {filename} ===")
        print(f"Old Label: {old_label} ({old_conf:.2%})")
        
        # Apply hybrid rules
        new_label, new_conf = apply_hybrid_rules(text, old_label, old_conf)
        print(f"New Predicted Label: {new_label} ({new_conf:.2%})")

if __name__ == "__main__":
    main()
