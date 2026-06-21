# Use a lightweight python runtime
FROM python:3.10-slim

# Install system dependencies (Tesseract OCR, Poppler-utils for PDF processing, build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install dependencies and download the SpaCy model
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY . .

# Expose port 7860 (default port expected by Hugging Face Spaces)
EXPOSE 7860

# Run the Streamlit app on port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
