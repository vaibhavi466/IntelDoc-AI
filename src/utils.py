import sqlite3
import datetime
import pandas as pd
import os
import fitz  
from src.config import DB_PATH

DB_NAME = str(DB_PATH)

#  PDF CONVERSION FUNCTION 
def convert_pdf_to_image(pdf_path, output_folder="data"):
    """
    Converts the first page of a PDF into a JPG image.
    Returns the path to the new image.
    """
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Load the first page (index 0)
        page = doc.load_page(0) 
        
        # Convert to image (pixmap)
        pix = page.get_pixmap(dpi=150)
        
        # Define new image path
        base_name = os.path.basename(pdf_path).replace(".pdf", ".jpg")
        image_path = os.path.join(output_folder, base_name)
        
        # Save the image
        pix.save(image_path)
        
        doc.close()
        return image_path
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None

# --- DATABASE FUNCTIONS ---
def init_db() -> None:
    """Initialize the SQLite database. Creates the documents table if it does not exist."""
    os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS documents
               (id             INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_date    TIMESTAMP,
                filename       TEXT,
                file_blob      BLOB,
                file_type      TEXT,
                category       TEXT,
                confidence     REAL,
                extracted_text TEXT,
                summary        TEXT)"""
        )

def save_to_db(uploaded_file, category: str, confidence: float, text: str, summary: str) -> str:
    """Save a processed document record to the database."""
    try:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                """INSERT INTO documents
                   (upload_date, filename, file_blob, file_type, category, confidence, extracted_text, summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.datetime.now(), uploaded_file.name, file_bytes,
                 uploaded_file.type, category, confidence, text, summary)
            )
        return "Document saved to Database!"
    except Exception as e:
        return f"DB Error: {e}"

def get_db_history() -> pd.DataFrame:
    """Return the full document history ordered by most recent first."""
    with sqlite3.connect(DB_NAME) as conn:
        return pd.read_sql_query(
            "SELECT id, upload_date, filename, category, confidence, summary "
            "FROM documents ORDER BY upload_date DESC",
            conn
        )
    

def delete_db_entries(ids_to_delete: list) -> bool:
    """Delete database rows by a list of IDs."""
    if not ids_to_delete:
        return False
    try:
        placeholders = ",".join(["?"] * len(ids_to_delete))
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                f"DELETE FROM documents WHERE id IN ({placeholders})",
                ids_to_delete
            )
        return True
    except Exception as e:
        print(f"Delete Error: {e}")
        return False

# --- TEXT METRIC FUNCTIONS ---
def calculate_text_metrics(text):
    if not text: return None
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_len = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Avg Word Length": round(avg_word_len, 1),
        "Readability Score (ARI)": round(4.71 * (len(text)/word_count) + 0.5 * (word_count/sentence_count) - 21.43, 1) if word_count > 0 and sentence_count > 0 else 0
    }


