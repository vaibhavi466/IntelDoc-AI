import streamlit as st
from transformers import pipeline
from src.utils import clean_ocr_text
from src.extraction import extract_information


@st.cache_resource
def _get_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # force CPU
    )

def generate_summary(text: str, category: str = "general")->str:
    """
    Summarizes document text. Uses structured summaries for standard document types
    and the abstractive model for text-heavy documents (news articles, letters).
    """
    category = category.lower().strip()
    
    # 1. Clean OCR text
    cleaned = clean_ocr_text(text)
    
    # If text is too short, avoid summarization model
    words = cleaned.split()
    if len(words) < 15:
        return "Document text is too brief to summarize."

    # 2. Category-Aware Smart Summarization
    if category == "invoice":
        info = extract_information(text, "invoice")
        vendor = info.get("Organizations", ["Unknown Vendor"])[0]
        inv_no = info.get("Invoice No.", "Unknown No.")
        amount = info.get("Total Amount", "Unknown Amount")
        date_str = info.get("Dates", ["Unknown Date"])[0]
        return f"Structured Summary: Invoice #{inv_no} from {vendor} dated {date_str} for a total of {amount}."

    elif category == "resume":
        info = extract_information(text, "resume")
        name = info.get("People/Names", ["Candidate"])[0]
        orgs = ", ".join(info.get("Organizations", []))
        orgs_str = f" with experience at {orgs}" if orgs else ""
        email = info.get("Emails", [""])[0]
        email_str = f" (Contact: {email})" if email else ""
        return f"Structured Summary: Professional resume of {name}{orgs_str}{email_str}."

    elif category == "email":
        info = extract_information(text, "email")
        subject = info.get("Subject", "No Subject")
        sender = info.get("People/Names", ["Sender"])[0]
        recipients = ", ".join(info.get("Organizations", []))
        rec_str = f" involving {recipients}" if recipients else ""
        return f"Structured Summary: Email message from {sender}{rec_str} regarding '{subject}'."

    elif category == "form":
        info = extract_information(text, "form")
        people = ", ".join(info.get("People/Names", []))
        orgs = ", ".join(info.get("Organizations", []))
        entity_str = f" containing details for {people or orgs}" if (people or orgs) else ""
        return f"Structured Summary: Standard form document{entity_str}."

    # 3. Fallback to Deep Learning Model for Letters and News Articles
    if len(words) < 40:
        return f"Short Document Summary: {cleaned}"
        
    summarizer = _get_summarizer()
    # Limit input size to fit model context (take first ~3000 chars)
    input_text = cleaned[:3000]

    try:
        summary_output = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
        return summary_output[0]['summary_text']
    except Exception as e:
        return f"Summary: {cleaned[:200]}..."