"""Load plain text from PDF, DOCX, TXT and MD files."""

from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_TYPES = (".pdf", ".docx", ".txt", ".md")


def _load_pdf(path: Path) -> str:
    # Extract text page by page; extract_text() can return None on empty pages.
    reader = PdfReader(path)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in pages if p)

    # No text at all usually means the PDF is just scanned images.
    if not text.strip():
        raise ValueError(
            f"{path.name} looks like a scanned PDF and has no extractable text."
        )
    return text


def _load_docx(path: Path) -> str:
    doc = Document(path)

    # Body paragraphs, skipping empty ones.
    parts = [p.text for p in doc.paragraphs if p.text.strip()]

    # Tables are stored separately from paragraphs, so read them too.
    # Each row becomes one line with cells separated by " | ".
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    return "\n\n".join(parts)


def _load_plain(path: Path) -> str:
    # errors="replace" swaps undecodable bytes for "?" instead of crashing.
    return path.read_text(encoding="utf-8", errors="replace")


def load_text(path) -> str:
    """Return the full text of a .pdf, .docx, .txt or .md file."""
    path = Path(path)
    ext = path.suffix.lower()

    # Pick the reader based on the file extension.
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext in (".txt", ".md"):
        return _load_plain(path)

    raise ValueError(
        f"Unsupported file type '{ext or '(none)'}' for {path.name}. "
        f"Supported types: {', '.join(SUPPORTED_TYPES)}"
    )
