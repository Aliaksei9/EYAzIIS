import fitz
from docx import Document
from striprtf.striprtf import rtf_to_text
from io import BytesIO


class FileHandler:
    def extract_text(self, uploaded_file):
        name = uploaded_file.name.lower()
        data = uploaded_file.getvalue()
        if name.endswith('.txt'):
            return data.decode('utf-8', errors='ignore')
        elif name.endswith('.pdf'):
            doc = fitz.open(stream=data, filetype="pdf")
            text = '\n'.join(page.get_text() or '' for page in doc)
            doc.close()
            return text
        elif name.endswith('.docx'):
            doc = Document(BytesIO(data))
            return '\n'.join(p.text for p in doc.paragraphs)
        elif name.endswith('.rtf') and rtf_to_text:
            return rtf_to_text(data.decode('utf-8', errors='ignore'))
        return None
