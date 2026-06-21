"""
EXPERIMENT — Early Prototype: EasyOCR Data Preparation (Superseded)
=====================================================================
This script was the initial data preparation pipeline using EasyOCR.
It was replaced by src/create_dataset.py which uses Tesseract (pytesseract)
for consistency with the production inference pipeline.

This file is kept to document the research process and compare OCR engines.
Do NOT use this in the main pipeline. Use src/create_dataset.py instead.

To run (requires: pip install easyocr):
    python experiments/01_data_prep.py
"""

import os
from tqdm import tqdm

try:
    import easyocr
    reader = easyocr.Reader(['en'])
except ImportError:
    easyocr = None
    reader = None

# --- CONFIG ---
# Path relative to: IntelDoc-AI/experiments/01_data_prep.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed_text")

def process_split(split_name):
    if reader is None:
        print("Error: easyocr reader is not initialized. Please run 'pip install easyocr' to use this experiment.")
        return
        
    source_path = os.path.join(RAW_DATA_DIR, split_name)
    target_path = os.path.join(OUTPUT_DIR, split_name)
    
    if not os.path.exists(source_path):
        print(f"Skipping {split_name} (Folder not found)")
        return

    # Get categories (email, invoice, resume, etc.)
    categories = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
    
    for category in categories:
        img_dir = os.path.join(source_path, category)
        txt_dir = os.path.join(target_path, category)
        os.makedirs(txt_dir, exist_ok=True)
        
        print(f"Processing {split_name}/{category}...")
        
        # Filter for images
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
        for img_file in tqdm(images):
            txt_filename = os.path.splitext(img_file)[0] + ".txt"
            save_path = os.path.join(txt_dir, txt_filename)
            
            if os.path.exists(save_path): continue
            
            try:
                result = reader.readtext(os.path.join(img_dir, img_file), detail=0)
                text_content = " ".join(result)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
            except Exception as e:
                print(f"Error on {img_file}: {e}")

if __name__ == "__main__":
    process_split("train")
    process_split("val")
    process_split("test")