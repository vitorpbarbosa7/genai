!pip install -q transformers einops accelerate bitsandbytes
!pip install -q langchain langchain_community langchain-huggingface langchainhub langchain_chroma

import torch
import os
import getpass

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["HF_TOKEN"] = getpass.getpass()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.1,
    max_new_tokens=500,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=pipe)

# PHI 3
#template = """
#<|system|>
#Você é um assistente virtual prestativo e está respondendo perguntas gerais. <|end|>
#<|user|>
#{pergunta}<|end|>
#<|assistant|>
#"""

# LLAMA 3
template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Você é um assistente virtual prestativo e está respondendo perguntas gerais.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{pergunta}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

template

prompt = PromptTemplate.from_template(template)
prompt

chain = prompt | llm

chain.invoke({"pergunta": "Que dia é hoje?"})

template_rag = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Você é um assistente virtual prestativo e está respondendo perguntas gerais.
Use os seguintes pedaços de contexto recuperado para responder à pergunta.
Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Pergunta: {pergunta}
Contexto: {contexto}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

prompt_rag = PromptTemplate.from_template(template_rag)
print(prompt_rag)

from datetime import date

dia = date.today()
print(dia)

contexto = "Você sabe que hoje é dia '{}'".format(dia)
print(contexto)

# Responda a pergunta com base apenas no contexto

chain_rag = prompt_rag | llm | StrOutputParser()

pergunta = "Que dia é hoje? Retorne a data em formato dd/mm/yyyy"

res = chain_rag.invoke({"pergunta": pergunta, "contexto": contexto})
res

prompt_rag

chain_rag = prompt_rag | llm | StrOutputParser()

contexto = """Faturamento trimestral:
1º: R$42476,40
2º: R$46212,97
3º: R$41324,56
4º: R$56430,24"""

#pergunta = "Qual é o faturamento do segundo trimestre?"
pergunta = "Qual trimestre teve o maior faturamento?"

chain_rag.invoke({
  "contexto": contexto,
  "pergunta": pergunta
})

from langchain.globals import set_debug
set_debug(True)

pergunta = "Qual trimestre teve o menor faturamento?"

chain_rag.invoke({
  "contexto": contexto,
  "pergunta": pergunta
})

set_debug(False)

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = WebBaseLoader(web_paths = ("https://www.bbc.com/portuguese/articles/cd19vexw0y1o",),)
docs = loader.load()

len(docs[0].page_content)

docs

print(docs[0].page_content[:300])

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, add_start_index = True)
splits = text_splitter.split_documents(docs)

len(splits)

splits[1]

splits[1].metadata

hf_embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

input_test = "Um teste apenas"
result = hf_embeddings.embed_query(input_test)

len(result)

print(result)

vectorstore = Chroma.from_documents(documents=splits, embedding=hf_embeddings)

# caso fossemos usar os embeddings da Open AI, basta mudar o método, passando direto conforme abaixo
#vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k": 6})

template_rag

#outra opção é importar o template de um prompt de RAG via hub.pull
#prompt = hub.pull("rlm/rag-prompt")

prompt_rag = PromptTemplate(
    input_variables=["contexto", "pergunta"],
    template=template_rag,
)
prompt_rag

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

chain_rag = ({"contexto": retriever | format_docs, "pergunta": RunnablePassthrough()}
             | prompt_rag
             | llm
             | StrOutputParser())

# Teste sem RAG
chain.invoke("Qual filme ganhou mais oscars na premiação de 2024?")

# Teste com RAG
chain_rag.invoke("Qual filme ganhou mais oscars na premiação de 2024?")

chain_rag.invoke("Quem ganhou o premio de melhor ator?")

vectorstore.delete_collection()
