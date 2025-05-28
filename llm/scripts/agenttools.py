!pip install -q transformers einops accelerate bitsandbytes
!pip install -q langchain langchain_community langchain-huggingface langchainhub langchain_chroma

!pip install wikipedia

from langchain_core.tools import Tool
from langchain.tools import WikipediaQueryRun, BaseTool
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000))

print(wikipedia.run("Deep Learning"))

print(wikipedia.run("Oscar Niemeyer"))

print(wikipedia.run("Oscars 2024"))

wikipedia_tool = Tool(name = "wikipedia",
                      description="You must never search for multiple concepts at a single step, you need to search one at a time. When asked to compare two concepts for example, search for each one individually.",
                      func=wikipedia.run)

def current_day(*args, **kwargs):
  from datetime import date

  dia = date.today()
  dia = dia.strftime('%d/%m/%Y')
  return dia

date_tool = Tool(name="Day", func = current_day,
                 description = "Use when you need to know the current date")

tools = [wikipedia_tool, date_tool]

import getpass
import os
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

from langchain import hub
prompt = hub.pull("hwchase17/react")

prompt

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

#llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1)

!pip install -q langchain-groq

from langchain_groq import ChatGroq
os.environ["GROQ_API_KEY"] = getpass.getpass()

llm = ChatGroq(model="llama3-70b-8192", temperature=0.7, max_tokens=None, timeout=None, max_retries=2)

from langchain.agents import AgentExecutor, create_react_agent

agent = create_react_agent(llm = llm, tools = tools, prompt = prompt)

agent_executor = AgentExecutor.from_agent_and_tools(agent = agent, tools = tools,
                                                    verbose = True, handling_parsing_errors = True)

print(prompt.template)

os.environ["HF_TOKEN"] = getpass.getpass()

resp = agent_executor.invoke({"input": "Em que dia estamos?"})

resp = agent_executor.invoke({"input": "Qual a população de Paris?"})

print(prompt.template)

#prompt.template = "Template personalizado aqui"

!pip install -q langchain-openai

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

agent = create_react_agent(
    llm=llm, tools=tools, prompt=prompt, stop_sequence=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

resp = agent_executor.invoke({"input": "Qual é a população de Paris?"})

os.environ["TAVILY_API_KEY"] = getpass.getpass()

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)

search_results = search.invoke("Qual é a populção de Paris?")
print(search_results)

search_results[0]['content']

from langchain_core.messages import HumanMessage
resp = llm.invoke([HumanMessage(content = "Olá! Tudo bem?")])

resp

tools

tools = [search]

tools

models_with_tools = llm.bind_tools(tools)

response = models_with_tools.invoke([HumanMessage(content = "Olá!")])
response

response.content

response.tool_calls

response = models_with_tools.invoke([HumanMessage(content="Qual é o valor de mercado atual da NVIDIA?")])

response.content, response.tool_calls

agent = create_react_agent(
    llm=llm, tools=tools, prompt=prompt, stop_sequence=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

resp = agent_executor.invoke({"input": "Qual é o valor de mercado recente da NVIDIA?"})

!pip install -q langgraph

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

from langgraph.prebuilt import create_react_agent as langgraph_react_agent

agent_executor = langgraph_react_agent(llm, tools)

input = "Qual é o valor de mercado recente da NVIDIA?"

response = agent_executor.invoke({"messages": [HumanMessage(content=input)]})
response["messages"]

response["messages"][-1]

response["messages"][-1].content
