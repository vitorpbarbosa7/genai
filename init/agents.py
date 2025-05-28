from langchain_openai import ChatOpenAI
import os

# 🔑 API setup
os.environ["OPENAI_API_KEY"] = os.getenv('ROUTER_API_KEY')
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 🧠 Agent 1: Philosopher
philosopher = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0  # Deterministic
)

# 🎭 Agent 2: Critic
critic = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0  # Deterministic
)

# 🚩 Step 1: Philosopher answers
question = "What is the meaning of life?"

philosopher_prompt = f"""
You are a philosopher. Answer this question with depth and clarity:

{question}
"""

philosopher_response = philosopher.invoke(philosopher_prompt)
print("🧠 Philosopher says:\n", philosopher_response.content)

# 🚩 Step 2: Critic evaluates the answer
critic_prompt = f"""
You are a sharp, skeptical critic. Here is what a philosopher said:

"{philosopher_response.content}"

Evaluate this answer. Is it profound, vague, naive, or insightful? Give an honest assessment.
"""

critic_response = critic.invoke(critic_prompt)
print("\n🎭 Critic replies:\n", critic_response.content)

