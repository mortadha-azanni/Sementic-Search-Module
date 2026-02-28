"""
Ingestion Pipeline — Processes PDFs from data/ into the embeddings table.

Uses the golden-chunk extractor (core.chunker) to produce high-quality
chunks, then embeds and stores them in PostgreSQL/pgvector.

Steps:
  1. Extract structured chunks from each PDF (core.chunker)
  2. Generate embeddings (all-MiniLM-L6-v2)
  3. Insert into PostgreSQL (pgvector)
"""

import os
from core.chunker import extract_chunks
from core.embedding import Embedder
from core.database import VectorDB


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Main Ingestion ────────────────────────────────────────────────────────

def ingest_all():
    """Process all PDFs in data/ and insert embeddings into DB."""
    print("=" * 60)
    print("  INGESTION PIPELINE (golden-chunk extractor)")
    print("=" * 60)

    # List PDFs
    pdf_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")])
    print(f"\n📁 Found {len(pdf_files)} PDF files in data/\n")

    if not pdf_files:
        print("❌ No PDF files found. Exiting.")
        return

    # Initialize components
    print("🔄 Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = Embedder()
    print("✅ Model loaded.\n")

    db = VectorDB()
    if not db.conn:
        print("❌ Database connection failed. Exiting.")
        return

    # Clear old data before re-ingesting
    print("🗑️  Clearing existing embeddings...")
    db.truncate_embeddings()
    print("✅ Table cleared.\n")

    total_chunks = 0
    skipped_dupes = 0
    errors = 0
    seen_chunks: set[str] = set()  # global dedup across all PDFs

    for doc_id, pdf_name in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(DATA_DIR, pdf_name)

        try:
            # 1. Extract structured chunks
            chunks = extract_chunks(pdf_path)

            if not chunks:
                print(f"  ⚠️  [{doc_id:02d}] {pdf_name} — no valid chunks, skipping")
                continue

            # 2. Deduplicate: skip chunks already seen in another PDF
            unique_chunks = []
            for chunk in chunks:
                if chunk not in seen_chunks:
                    seen_chunks.add(chunk)
                    unique_chunks.append(chunk)
                else:
                    skipped_dupes += 1

            if not unique_chunks:
                print(f"  ⚠️  [{doc_id:02d}] {pdf_name} — all chunks are duplicates, skipping")
                continue

            # 3. Generate embeddings
            vectors = [embedder.encode_query(chunk) for chunk in unique_chunks]

            # 4. Insert into DB
            rows = [(doc_id, chunk, vec) for chunk, vec in zip(unique_chunks, vectors)]
            db.insert_embeddings_batch(rows)

            total_chunks += len(unique_chunks)
            dupe_note = f" ({len(chunks) - len(unique_chunks)} dupes skipped)" if len(unique_chunks) < len(chunks) else ""
            print(f"  ✅  [{doc_id:02d}] {pdf_name} — {len(unique_chunks)} chunks inserted{dupe_note}")

        except Exception as e:
            errors += 1
            print(f"  ❌  [{doc_id:02d}] {pdf_name} — ERROR: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  DONE — {total_chunks} unique chunks from {len(pdf_files)} PDFs")
    if skipped_dupes:
        print(f"  🔁 {skipped_dupes} duplicate chunks skipped")
    if errors:
        print(f"  ⚠️  {errors} files had errors")
    print(f"  📊 Database now has {db.count_embeddings()} rows")
    print(f"{'=' * 60}")

    db.close()


if __name__ == "__main__":
    ingest_all()
