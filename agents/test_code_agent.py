from agents.code_agent import CodeAgent


agent = CodeAgent()


query = input("\nAsk the Code Agent: ")


answer = agent.answer(query)


print("\n==============================")
print("CODE AGENT RESPONSE")
print("==============================\n")

print(answer)