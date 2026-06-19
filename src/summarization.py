import streamlit as st
from transformers import pipeline


@st.cache_resource
def _get_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # force CPU
    )

def generate_summary(text: str)->str:
    """
    Summarizes long document text into a short paragraph.
    """
    # 1. Safety Check: If text is too short, don't summarize
    if len(text.split()) < 50:
        return "Document is too short to summarize."
    summarizer = _get_summarizer()

    # 2. Chunking (Handling long documents)
    # Models have a limit (usually 1024 tokens). We take the first ~3000 chars.
    # For a portfolio project, summarizing the first page is usually enough.
    max_input_length = 3000 
    input_text = text[:max_input_length]

    try:
        summary_output = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
        
        return summary_output[0]['summary_text']
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"