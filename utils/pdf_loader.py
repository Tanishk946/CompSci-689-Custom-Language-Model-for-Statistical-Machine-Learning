import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    text_pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text().strip()
            print(text)
            if text:
                text_pages.append(text)
    return "\n\n".join(text_pages)

def load_all_pdfs(pdf_folder):
    full_text = ""
    for fname in os.listdir(pdf_folder):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, fname)
            full_text += extract_text_from_pdf(path) + "\n\n"
    return full_text

load_all_pdfs("G:/Masters/ML 689/689LM/data/lectures")  # Example usage
