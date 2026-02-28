import json
import os
import sys
from core.embedding import Embedder
from core.database import VectorDB

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def save_results(question, results):
    """Save search results to output/last_results.json."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = {
        "question": question,
        "results": [
            {
                "rank": i,
                "id_document": res["id_document"],
                "texte_fragment": res["texte_fragment"],
                "similarity": float(res["similarity"]),
            }
            for i, res in enumerate(results, 1)
        ],
    }
    path = os.path.join(OUTPUT_DIR, "last_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("--- BAKERY/PASTRY FORMULATION ASSISTANT ---")

    # Initialize embedding model
    try:
        embedder = Embedder()
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        sys.exit(1)

    # Connect to database
    try:
        db = VectorDB()
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)

    if not db.conn:
        print("❌ No database connection. Check your .env credentials.")
        sys.exit(1)

    with db:
        count = db.count_embeddings()
        if count == 0:
            print("⚠️  Database is empty. Run 'uv run ingest.py' first.\n")
        else:
            print(f"📊 Database contains {count} fragments.\n")

        while True:
            try:
                question = input("[?] Ask about ingredients (or 'exit'): ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            question = question.strip()
            if not question:
                continue
            if question.lower() == "exit":
                break

            # 1. Generate Embedding
            query_vector = embedder.encode_query(question)

            # 2. Search & Rank in DB
            results = db.search_similar_fragments(query_vector, top_k=3)

            if not results:
                print("\nNo results found.\n")
                continue

            # 3. Display Results (matching spec format)
            print()
            for i, res in enumerate(results, 1):
                print(f"Résultat {i}")
                print(f'Texte: "{res["texte_fragment"]}"')
                print(f"Score: {res['similarity']:.2f}")
                print()

            # 4. Save results to output/
            save_results(question, results)


if __name__ == "__main__":
    main()
