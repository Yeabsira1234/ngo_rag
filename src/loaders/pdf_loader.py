import fitz


class PDFLoader:

    def load(self, pdf_path: str) -> str:
        """
        Load a PDF and return all text.
        """

        document = fitz.open(pdf_path)

        text = ""

        for page in document:
            text += page.get_text()

        document.close()

        return text