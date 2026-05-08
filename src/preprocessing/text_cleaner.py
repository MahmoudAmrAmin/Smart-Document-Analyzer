import re 

def clean_text(text :str) : 
    """
        Clean extracted PDF text:
        - Remove excessive whitespace and newlines
        -Remove page markers we added during ingestion
        - Strip non-printable characters
        - Normalize spacing
    """
     
    # Remove our page markers from pdf_loader
    text = re.sub(r"--- Page \d+ .*?---", "", text)

    # Remove non-printable / control characters (keep newlines for now)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_documents(documents):
    """
    Apply clean_text to a dict of {filename: raw_text}
    Returns {filename: clean_text}
    """
    cleaned = {}
    for filename, raw_text in documents.items():
        cleaned[filename] = clean_text(raw_text)
        print(f"Cleaned: {filename} — {len(cleaned[filename])} chars")
    return cleaned