import chromadb
import numpy as np
from typing import List


class VectorStore:

    def __init__(self, embedding_manager, collection_name: str = "waste_management"):

        self.embedding_manager = embedding_manager

        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray
    ):
        if len(texts) != len(embeddings):
            raise ValueError(
                "Number of texts must match number of embeddings"
            )

        ids = [f"chunk_{i}" for i in range(len(texts))]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist()
        )

        print(f"Stored {len(texts)} chunks in ChromaDB")


    def search(self, query: str, n_results: int = 3):
        """
        Retrieve relevant document chunks for a query.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Generate embedding for the user query
        query_embedding = self.embedding_manager.generate_embeddings(
            [query]
        )

        # Retrieve relevant chunks from ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(n_results, self.collection.count())
        )

        return results