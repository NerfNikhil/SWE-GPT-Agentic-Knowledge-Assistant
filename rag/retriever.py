import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(self, documents):

        self.documents = documents

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        embeddings = self.model.encode(documents)

        embeddings = np.array(
            embeddings
        ).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)


    def search(self, query, k=3):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(
            query_embedding
        ).astype("float32")
        distance, indices = self.index.search(
            query_embedding,
            k
        )
        results = []
        for i in indices[0]:
            results.append(self.documents[i])
        return results