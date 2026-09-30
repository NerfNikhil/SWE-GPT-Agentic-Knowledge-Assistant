from transformers import pipeline


class HRAgent:

    def __init__(self):

        self.llm = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-360M-Instruct"
        )

        self.hr_knowledge = """
        Employees should complete onboarding documentation
        and required training during the onboarding process.

        Employees should follow company policies regarding
        leave, attendance, conduct, and workplace behavior.

        HR is responsible for supporting employees with
        workplace policies, benefits, onboarding, and
        employee-related concerns.
        """


    def answer(self, query):

        prompt = f"""
You are an HR assistant.

Use the following HR knowledge to answer the question.

HR Knowledge:
{self.hr_knowledge}

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