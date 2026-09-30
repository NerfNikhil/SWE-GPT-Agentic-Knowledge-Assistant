from transformers import pipeline

from rag.retriever import Retriever
from rag.loader import load_document, chunk_text


class CodeAgent:

    def __init__(self):

        # -------------------------
        # Load knowledge
        # -------------------------

        text = load_document("data/python.txt")

        documents = chunk_text(
            text,
            chunk_size=500
        )

        self.retriever = Retriever(documents)

        # -------------------------
        # Load LLM
        # -------------------------

        self.llm = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-360M-Instruct"
        )


    def answer(self, query):

        # -------------------------
        # Retrieve context
        # -------------------------

        results = self.retriever.search(
            query,
            k=3
        )

        context = "\n\n".join(results)

        # -------------------------
        # Build agent prompt
        # -------------------------

        prompt = f"""
You are a software engineering expert.

Answer the user's question using the
provided context.

If the user asks for code, provide
correct and clear code.

Context:
{context}

User Question:
{query}

Answer:
"""

        # -------------------------
        # Generate answer
        # -------------------------

        response = self.llm(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3
        )

        return response[0]["generated_text"]