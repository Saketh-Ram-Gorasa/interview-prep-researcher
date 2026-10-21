"""Resume text from an uploaded PDF. Resume data stays in the run and never enters the shared corpus."""

import io

from pypdf import PdfReader


def read_pdf(data: bytes) -> str:
    """Text of a text-based PDF; scanned PDFs return little or nothing (OCR is out of scope)."""
    pages = PdfReader(io.BytesIO(data)).pages
    return "\n".join(page.extract_text() or "" for page in pages).strip()
