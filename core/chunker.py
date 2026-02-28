"""
Golden-Chunk Extractor — Section-aware, table-aware, product-name-prefixed
chunking for bakery/pastry technical data sheets (PDF).

Produces high-quality text chunks from 2-page technical sheets:
  - Page 1: product description, effective material, dosage, etc.
  - Page 2: microbiology, allergens, packaging, storage, etc.
  - Extra pages: generic paragraph-based splitting.
"""

import re
from PyPDF2 import PdfReader

# ── Config ────────────────────────────────────────────────────────────────

MAX_CHARS = 800
MIN_CHARS = 20

HEADER_NOISE = [
    r"^VTR&beyond$", r"^No\.\d+,", r"^Zone,",
    r"^Stresemannstr", r"^Tel:", r"^Mail:", r"^Website:",
]

PAGE1_SECTIONS = {
    "ProductDescription", "Product Description",
    "Effectivematerial", "Effective material",
    "Activity", "Application", "Function", "Dosage",
    "Organoleptic", "Physicochemical",
    "BakeryEnzyme", "Bakery Enzyme",
}

PAGE2_SECTIONS = [
    "Microbiology", "Heavymetals", "Heavy metals",
    "GMOstatus", "Ionizationstatus", "Ionization status",
    "Allergens", "Packaging", "Storage",
]

PAGE2_SKIP = {"FOODSAFTYDATA", "FOODSAFETY DATA", "FOOD SAFETY DATA"}


# ── Helpers ───────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """Remove PDF artifacts and normalise whitespace."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_header(text: str) -> list[str]:
    """Return content lines with header noise removed."""
    return [
        ln.strip() for ln in text.splitlines()
        if ln.strip() and not any(re.match(p, ln.strip()) for p in HEADER_NOISE)
    ]


def split_long(chunk: str) -> list[str]:
    """Split a chunk exceeding MAX_CHARS on sentence boundaries."""
    if len(chunk) <= MAX_CHARS:
        return [chunk]
    parts, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", chunk):
        candidate = f"{cur} {sentence}".strip() if cur else sentence
        if len(candidate) <= MAX_CHARS:
            cur = candidate
        else:
            if cur:
                parts.append(cur)
            cur = sentence
    if cur:
        parts.append(cur)
    return parts


def emit(chunks: list[str], prefix: str, label: str, body: str):
    """Build a prefixed chunk and append valid pieces to *chunks*."""
    tag = f"{label}: " if label else ""
    for piece in split_long(f"{prefix} — {tag}{body}"):
        if len(piece) >= MIN_CHARS:
            chunks.append(piece)


# ── Page processors ──────────────────────────────────────────────────────

def page1_chunks(lines: list[str], name: str) -> list[str]:
    """Group lines into section paragraphs, prefix with product name."""
    chunks, buf = [], []
    skip = {"TECHNICALDATASHEET", "TECHNICAL DATA SHEET", name}

    for ln in lines:
        if ln in skip:
            continue
        if ln in PAGE1_SECTIONS:
            if buf:
                emit(chunks, name, "", " ".join(buf))
                buf.clear()
            buf.append(f"{ln}:")
        else:
            buf.append(ln)

    if buf:
        emit(chunks, name, "", " ".join(buf))
    return chunks


def page2_chunks(lines: list[str], name: str) -> list[str]:
    """Group lines by section header, one chunk per section."""
    chunks, buf, section = [], [], ""

    def flush():
        nonlocal section
        if buf:
            emit(chunks, name, section, " | ".join(buf))
            buf.clear()

    for ln in lines:
        if ln in PAGE2_SKIP:
            continue

        matched = next(
            (s for s in PAGE2_SECTIONS if ln == s or ln.startswith(s)),
            None,
        )

        if matched:
            flush()
            section = matched
            remainder = ln[len(matched):].strip()
            if remainder:
                buf.append(remainder)
        else:
            buf.append(ln)

    flush()
    return chunks


def generic_chunks(lines: list[str], name: str) -> list[str]:
    """Fallback: split on blank-line boundaries."""
    chunks = []
    for paragraph in "\n".join(lines).split("\n\n"):
        if paragraph.strip():
            emit(chunks, name, "", paragraph.strip())
    return chunks


# ── PDF → Chunks ──────────────────────────────────────────────────────────

def get_product_name(lines: list[str]) -> str:
    """Find the product name (first line after 'TECHNICAL DATA SHEET')."""
    for i, line in enumerate(lines):
        if re.search(r"technical\s*data\s*sheet", line, re.I) and i + 1 < len(lines):
            return lines[i + 1]
    return lines[0] if lines else "Unknown Product"


def extract_chunks(pdf_path: str) -> list[str]:
    """Read a PDF and return a list of golden chunks."""
    reader = PdfReader(pdf_path)
    pages = [strip_header(clean(p.extract_text() or "")) for p in reader.pages]

    name = get_product_name(pages[0])

    chunks = page1_chunks(pages[0], name)
    if len(pages) >= 2:
        chunks += page2_chunks(pages[1], name)
    for p in pages[2:]:
        chunks += generic_chunks(p, name)

    return chunks
