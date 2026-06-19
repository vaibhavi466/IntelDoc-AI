import re
import spacy

# Load the NLP model once as a global variable.
# "en_core_web_sm" is a small English model trained on web text.
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

    Args:
        text (str): Extracted document text.
        category (str): Predicted document category.

    Returns:
        dict: Extracted key information.
    """

    category = category.lower().strip()
    results = {}

    # Safety check
    if not text or not text.strip():
        return results

    # --- 1. SPACY NER EXTRACTION ---

    doc = nlp(text)

    people = list(set(ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"))
    orgs = list(set(ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"))

    # Filter out short/noisy entities
    people = [p for p in people if len(p) > 2]
    orgs = [o for o in orgs if len(o) > 2]

    if people:
        results["People/Names"] = people[:5]

    if orgs:
        results["Organizations"] = orgs[:5]

    # --- 2. COMMON REGEX PATTERNS ---

    text_lines = text.split("\n")

    # Common date pattern used by invoice, letter, and form
    dates = re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)

    # Common phone pattern used by resume and form
    phone_pattern = r"(?:\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
    phones = re.findall(phone_pattern, text)

    # Common email pattern used by email and resume
    emails = re.findall(
        r"[a-zA-Z0-9._%+\-]+(?:@|©)[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        text,
    )
    clean_emails = [
        email.replace(" ", "").replace("©", "@")
        for email in emails
        if "@" in email.replace("©", "@")
    ]

    # --- 3. EMAIL CATEGORY SPECIFICS ---

    if category in ("email", "resume"):
        if clean_emails:
            results["Emails"] = list(set(clean_emails))

        if category == "email":
            for line in text_lines:
                line_clean = line.strip()
                if line_clean.lower().startswith(("subject:", "re:")):
                    results["Subject"] = line_clean
                    break

    # --- 4. INVOICE CATEGORY SPECIFICS ---

    if category == "invoice":
        amounts = re.findall(r"\$\s?([0-9,]+\.[0-9]{2})", text)

        if amounts:
            try:
                floats = [float(amount.replace(",", "")) for amount in amounts]
                results["Total Amount"] = f"${max(floats):,.2f}"
            except (ValueError, TypeError):
                pass

        if dates:
            results["Dates"] = dates[:3]

        invoice_no = re.findall(
            r"(?:Invoice|INV)[#:\s]+([A-Z0-9\-]+)",
            text,
            re.IGNORECASE,
        )

        if invoice_no:
            results["Invoice No."] = invoice_no[0]

    # --- 5. RESUME CATEGORY SPECIFICS ---

    if category == "resume":
        if phones:
            results["Phone"] = phones[0].strip()

        linkedin = re.findall(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)

        if linkedin:
            results["LinkedIn"] = linkedin[0]

    # --- 6. LETTER CATEGORY SPECIFICS ---

    if category == "letter":
        if dates:
            results["Date"] = dates[0]

        for line in text_lines:
            line_clean = line.strip()
            if line_clean.lower().startswith(("dear ", "to whom", "re:")):
                results["Salutation"] = line_clean
                break

    # --- 7. FORM CATEGORY SPECIFICS ---

    if category == "form":
        if dates:
            results["Dates"] = dates[:3]

        if phones:
            results["Phone"] = phones[0].strip()

    return results