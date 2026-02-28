from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        # REQUIRED MODEL: all-MiniLM-L6-v2
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)

    def encode_query(self, query: str):
        """Converts user question into a vector."""
        return self.model.encode(query).tolist()

