from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


text1 = "Binary search is an efficient searching algorithm."

text2 = "Binary search works by repeatedly dividing a sorted array."

text3 = "Python dictionaries store key-value pairs."


embedding1 = model.encode([text1])
embedding2 = model.encode([text2])
embedding3 = model.encode([text3])


print("Similarity between text1 and text2:")
print(cosine_similarity(embedding1, embedding2)[0][0])


print("\nSimilarity between text1 and text3:")
print(cosine_similarity(embedding1, embedding3)[0][0])