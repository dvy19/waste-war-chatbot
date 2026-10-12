import chromadb
import numpy as np
from typing import List


class VectorStore:

    def __init__(self, collection_name: str = "waste_management"):

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