import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


documents = [
    "Python supports several data structures including lists, tuples, sets, and dictionaries.",
    "A list is an ordered and mutable collection of elements.",
    "A dictionary stores data in key-value pairs.",
    "Binary search is an efficient searching algorithm that works on sorted arrays.",
    "Binary search repeatedly divides the search space into two halves.",
    "The time complexity of binary search is O(log n)."
]


# Convert documents into embeddings
embeddings = model.encode(documents)

# Convert to numpy float32
embeddings = np.array(embeddings).astype("float32")


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
index.add(embeddings)

print("Number of vectors:", index.ntotal)
query = "How does binary search work?"

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")


distance, indices = index.search(query_embedding, k=3)


print("\nQuery:", query)

for i in range(3):
    print("\nResult:", i + 1)
    print("Distance:", distance[0][i])
    print("Document:", documents[indices[0][i]])