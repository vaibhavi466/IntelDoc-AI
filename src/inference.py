"""
inference.py — Document prediction pipeline for DocuMind AI.

Performs OCR on a document image and classifies it using the
fine-tuned DistilBERT model. Requires a running Streamlit context
for @st.cache_resource to function.
"""
import streamlit as st  
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification  
import pytesseract  #Tesseract → extract text from images
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
    
    return label, confidence.item(), text

if __name__ == "__main__":
    # Test with a dummy path
    print("Inference script ready. Run via App.")