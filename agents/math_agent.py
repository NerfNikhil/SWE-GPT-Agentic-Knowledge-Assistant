from transformers import pipeline


class MathAgent:

    def __init__(self):

        self.llm = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-360M-Instruct"
        )


    def answer(self, query):

        prompt = f"""
You are a mathematics expert.

Solve the user's mathematical question
carefully and explain the reasoning clearly.

User Question:
{query}

Answer:
"""

        response = self.llm(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.2
        )

        return response[0]["generated_text"]