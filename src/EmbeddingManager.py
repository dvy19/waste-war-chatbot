from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully")



class EmbeddingManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of text chunks

        Returns:
            NumPy array containing embeddings
        """

        if not texts:
            raise ValueError("texts cannot be empty")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        return np.array(embeddings)
