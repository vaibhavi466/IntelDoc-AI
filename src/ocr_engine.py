import os
import sys
import pytesseract
from PIL import Image

_tesseract_path = os.getenv("TESSERACT_CMD")
if _tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_path

def extract_text(image_path: str) -> str:
    """
    Reads an image from the path and returns the text string.
    """
    try:
        # 1. Load the image
        img = Image.open(image_path)
        
        # 2. Convert to text using Tesseract
        text = pytesseract.image_to_string(img)
        
        return text
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""
    
if __name__ == "__main__":
    
    
    # Update this path to point to a REAL file in the data folder
    test_image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "raw", "train", "invoice", "sample.tif")
    
    print(f"Testing OCR on: {test_image_path}")
    
    # Run the function
    result = extract_text(test_image_path)
    
    print("-" * 30)
    print("EXTRACTED TEXT:")
    print("-" * 30)
    print(result)