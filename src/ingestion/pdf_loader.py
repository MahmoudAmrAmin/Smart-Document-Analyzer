import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    Tries direct text extraction first (fast).
    Falls back to OCR if no text is found (scanned PDFs).
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        # Try direct text extraction
        text = page.get_text()

        if text.strip():
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        else:
            # Scanned page — use OCR
            print(f"  Page {page_num + 1}: no text found, running OCR...")
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(image)
            full_text += f"\n--- Page {page_num + 1} (OCR) ---\n{ocr_text}"

    doc.close()
    return full_text


def load_documents_from_folder(folder_path: str) -> dict:
    """
    Load all PDFs from a folder.
    Returns a dict: { filename: extracted_text }
    """
    results = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in folder.")
        return results

    for filename in pdf_files:
        path = os.path.join(folder_path, filename)
        print(f"Loading: {filename}")
        text = extract_text_from_pdf(path)
        results[filename] = text
        print(f"  → Extracted {len(text)} characters\n")

    return results

