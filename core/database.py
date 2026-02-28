import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

class VectorDB:
    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                port=os.getenv("DB_PORT")
            )
            print("✅ Successfully connected to Rose Blanche PostgreSQL.")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.conn = None

    def insert_embedding(self, id_document, texte_fragment, vecteur):
        """Insert a single embedding row into the database."""
        if not self.conn:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO embeddings (id_document, texte_fragment, vecteur) VALUES (%s, %s, %s::vector)",
                (id_document, texte_fragment, str(vecteur))
            )
        self.conn.commit()

    def insert_embeddings_batch(self, rows):
        """Insert multiple (id_document, texte_fragment, vecteur) rows."""
        if not self.conn:
            return
        with self.conn.cursor() as cur:
            for id_doc, text, vec in rows:
                cur.execute(
                    "INSERT INTO embeddings (id_document, texte_fragment, vecteur) VALUES (%s, %s, %s::vector)",
                    (id_doc, text, str(vec))
                )
        self.conn.commit()

    def truncate_embeddings(self):
        """Delete all rows and reset the id sequence."""
        if not self.conn:
            return
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE embeddings RESTART IDENTITY")
        self.conn.commit()

    def count_embeddings(self):
        """Return total row count in embeddings table."""
        if not self.conn:
            return 0
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings;")
            return cur.fetchone()[0]

    def search_similar_fragments(self, query_vector, top_k=3):
        """
        Executes the Cosine Similarity search inside PostgreSQL.
        The <=> operator calculates Cosine Distance. 
        Similarity = 1 - Distance.
        """
        if not self.conn:
            return []

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            search_query = """
                SELECT id_document, texte_fragment, 1 - (vecteur <=> %s::vector) AS similarity
                FROM embeddings
                ORDER BY similarity DESC
                LIMIT %s;
            """
            cur.execute(search_query, (str(query_vector), top_k))
            return cur.fetchall()

    def close(self):
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
