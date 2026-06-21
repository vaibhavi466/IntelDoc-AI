import re
import spacy
from src.utils import clean_ocr_text

# Load the NLP model once as a global variable.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Downloading it now...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_information(text: str, category: str) -> dict:
    """
    Extracts important information from document text based on predicted category.
    Cleans text first and filters out noisy entities.
    """
    category = category.lower().strip()
    results = {}

    if not text or not text.strip():
        return results

    # 1. Clean the text for NER
    cleaned_text = clean_ocr_text(text)
    doc = nlp(cleaned_text)

    # 2. Extract and filter NER entities
    raw_people = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
    raw_orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]

    # Filter function to remove OCR noise
    def is_valid_entity(ent_text):
        if len(ent_text) < 3 or len(ent_text) > 40:
            return False
        # Remove entities that are just numbers/dates/punctuation
        if re.match(r'^[0-9\s.,\-\/#():]+$', ent_text):
            return False
        # Filter out common OCR noise / generic UI words
        noise_words = {
            'screenshot', 'click', 'search', 'page', 'image', 'menu', 'button', 
            'user', 'profile', 'friday', 'monday', 'tuesday', 'wednesday', 
            'thursday', 'saturday', 'sunday', 'january', 'february', 'march', 
            'april', 'may', 'june', 'july', 'august', 'september', 'october', 
            'november', 'december'
        }
        if ent_text.lower() in noise_words:
            return False
        return True

    people = list(set(p for p in raw_people if is_valid_entity(p)))
    orgs = list(set(o for o in raw_orgs if is_valid_entity(o)))

    # Deduplicate substrings: if "Google" and "Google Inc." both exist, keep "Google Inc."
    def remove_substrings(ent_list):
        sorted_ents = sorted(ent_list, key=len, reverse=True)
        keep = []
        for ent in sorted_ents:
            if not any(ent in other for other in keep):
                keep.append(ent)
        return keep

    people = remove_substrings(people)
    orgs = remove_substrings(orgs)

    if people:
        results["People/Names"] = people[:5]
    if orgs:
        results["Organizations"] = orgs[:5]

    # 3. Enhanced Date Extraction
    # Match dd/mm/yyyy, dd.mm.yyyy, dd-mm-yyyy and Month dd, yyyy
    date_patterns = [
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b"
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    dates = list(set(dates))
    if dates:
        results["Dates"] = dates[:3]

    # 4. Email Extraction
    emails = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", cleaned_text)
    if emails:
        results["Emails"] = list(set(emails))[:3]

    # 5. Phone Extraction
    phone_pattern = r"\b(?:\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
    phones = re.findall(phone_pattern, cleaned_text)
    if phones:
        results["Phones"] = list(set(phones))[:2]

    # 6. Category-specific extraction details
    if category == "invoice":
        # Look for invoice numbers
        inv_no = re.findall(r"(?:invoice|inv|bill|invoice\s*#|inv\s*#|no\.)[#:\s]*([A-Z0-9\-]{4,})", text, re.IGNORECASE)
        if inv_no:
            results["Invoice No."] = inv_no[0]
        
        # Look for amounts: $X,XXX.XX or €X,XXX.XX or just XXX.XX near total/amount keywords
        amounts = re.findall(r"[\$\u20AC\u00A3]\s*([0-9,]+\.[0-9]{2})", text)
        if not amounts:
            # If no currency symbol, look for numbers near "total", "amount", "due", "payment"
            matches = re.finditer(r"(?:total|due|amount|payment|subtotal|balance)\s*[:\-\s]*([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
            amounts = [m.group(1) for m in matches]
        
        if amounts:
            try:
                floats = [float(amount.replace(",", "")) for amount in amounts]
                results["Total Amount"] = f"${max(floats):,.2f}"
            except (ValueError, TypeError):
                pass

    elif category == "resume":
        linkedin = re.findall(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)
        if linkedin:
            results["LinkedIn"] = linkedin[0]
            
        git = re.findall(r"github\.com/[\w\-]+", text, re.IGNORECASE)
        if git:
            results["GitHub"] = git[0]

    elif category == "email":
        subject = None
        for line in text.split("\n"):
            if line.strip().lower().startswith(("subject:", "re:", "fwd:")):
                subject = line.strip()
                break
        if subject:
            results["Subject"] = subject

    return results