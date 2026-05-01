import os, fitz
from logs.logger import log

class PDFIngestor:
    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def list_pdfs(self):
        return sorted(f for f in os.listdir(self.folder) if f.lower().endswith(".pdf"))

    def load(self, filename):
        path = os.path.join(self.folder, filename)
        doc = fitz.open(path)
        return list(doc)
