from pdf_loader import load_documents_from_folder

docs = load_documents_from_folder(r"M:\Machine learning carrer\Projects\End-to-end Systems\Smart Document analyzer\data\raw\documents")

for name, text in docs.items():
    print(f"\n{'='*40}")
    print(f"Document: {name}")
    print(f"Characters extracted: {len(text)}")
    print(f"Preview:\n{text[:300]}")