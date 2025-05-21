from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = os.getenv('ROUTER_API_KEY')
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 🎯 The Main Agent (Philosopher)
agent = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0
)

# 🔐 The Guard Agent (Semantic Filter)
guard = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0
)

# 🚩 Check if the question is valid
def guard_check(user_input):
    prompt = f"""
You are a strict safety and domain guard for an AI assistant.

Check the following user question.

If the question is:
- Dangerous (violence, hacking, harm, malware, weapons, illegal activity), OR
- Off-topic (math, programming, fitness, politics, non-philosophical things),

Answer only with "Reject".

Otherwise, if it's safe and philosophical, answer with "Accept".

Question:
{user_input}
"""
    result = guard.invoke(prompt).content.strip().lower()
    return result


# 🔥 Main function
def run_agent(user_input):
    check = guard_check(user_input)
    if "reject" in check:
        return "❌ Your question violates the assistant's safety or domain rules."

    # Safe → run agent
    main_prompt = f"""
You are a philosopher. Only answer philosophical questions.
If the question is not about philosophy, say: "I'm a philosophy assistant and cannot answer that."

Question:
{user_input}
"""
    response = agent.invoke(main_prompt)
    return response.content


# 🚀 Test Examples
print(run_agent("Explain the concept of existentialism."))
print(run_agent("How to hack a server?"))
print(run_agent("What's the derivative of x squared?"))

