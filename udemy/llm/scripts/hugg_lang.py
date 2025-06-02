!pip install -q transformers==4.48.2 einops accelerate bitsandbytes

#!pip install bitsandbytes-cuda110 bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

import torch
import getpass
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

device

torch.random.manual_seed(42)

os.environ["HF_TOKEN"] = getpass.getpass()

# você também pode deixar escrito direto no código para facilitar reexecuções futuras
# só tome cuidado se for compartilhar seu código em algum local, pois você nunca deve deixar suas chaves expostas (principalmente se for de uma API paga)
#os.environ["HF_TOKEN"] = "hf_..."

id_model = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    id_model,
    device_map = "cuda",
    torch_dtype = "auto",
    trust_remote_code = True,
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained(id_model)

pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.1, # 0.1 até 0.9
    "do_sample": True,
}

prompt = "Explique o que é computação quântica"
#prompt = "Quanto é 7 x 6 - 42?"

output = pipe(prompt, **generation_args)

output

print(output[0]['generated_text'])

prompt = "Quanto é 7 x 6 - 42?"
output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])

prompt = "Quem foi a primeira pessoa no espaço?"
output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

template

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

prompt = "Você entende português?"

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

prompt = "O que é IA?"  # @param {type:"string"}

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

#prompt = "O que é IA? "  # @param {type:"string"}
#prompt = "O que é IA? Responda em 1 frase" # @param {type:"string"}
prompt = "O que é IA? Responda em forma de poema" # @param {type:"string"}

sys_prompt = "Você é um assistente virtual prestativo. Responda as perguntas em português."

template = """<|system|>
{}<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(sys_prompt, prompt)

print(template)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

prompt = "Gere um código em python que escreva a sequência de fibonnaci"

sys_prompt = "Você é um programador experiente. Retorne o código requisitado e forneça explicações breves se achar conveniente"

template = """<|system|>
{}<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(sys_prompt, prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

def fibonacci(n):

    a, b = 0, 1

    sequence = []

    while len(sequence) < n:

        sequence.append(a)

        a, b = b, a + b

    return sequence


# Exemplo de uso:

n = 10  # Quantidade de números da sequência de Fibonacci a serem gerados

print(fibonacci(n))

prompt = "O que é IA?"

msg = [
    {"role": "system", "content": "Você é um assistente virtual prestativo. Responda as perguntas em português."},
    {"role": "user", "content": prompt}
]

output = pipe(msg, **generation_args)
print(output[0]["generated_text"])

prompt = "Liste o nome de 10 cidades famosas da Europa"
prompt_sys = "Você é um assistente de viagens prestativo. Responda as perguntas em português."

msg = [
    {"role": "system", "content": prompt_sys},
    {"role": "user", "content": prompt},
]

output = pipe(msg, **generation_args)
print(output[0]['generated_text'])

!nvidia-smi

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = ("Quem foi a primeira pessoa no espaço?")
messages = [{"role": "user", "content": prompt}]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
model_inputs = encodeds.to(device)
generated_ids = model.generate(model_inputs, max_new_tokens = 1000, do_sample = True,
                               pad_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(generated_ids)
res = decoded[0]
res

!pip install -q langchain
!pip install -q langchain-community
!pip install -q langchain-huggingface
!pip install -q langchainhub
!pip install -q langchain_chroma

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

model_id = "microsoft/Phi-3-mini-4k-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    model = model,
    tokenizer = tokenizer,
    task = "text-generation",
    temperature = 0.1,
    max_new_tokens = 500,
    do_sample = True,
    repetition_penalty = 1.1,
    return_full_text = False,
)

llm = HuggingFacePipeline(pipeline = pipe)

input = "Quem foi a primeira pessoa no espaço?"

output = llm.invoke(input)
print(output)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
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

input = "Qual foi a primeira linguagem de programação?"

output = llm.invoke(input)

print(output)

template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

system_prompt = "Você é um assistente e está respondendo perguntas gerais."
user_prompt = input

prompt_template = template.format(system_prompt = system_prompt, user_prompt = user_prompt)
prompt_template

output = llm.invoke(prompt_template)
output

from langchain_core.messages import (HumanMessage, SystemMessage)
from langchain_huggingface import ChatHuggingFace

msgs = [
    SystemMessage(content = "Você é um assistente e está respondendo perguntas gerais."),
    HumanMessage(content = "Explique para mim brevemente o conceito de IA.")
]

chat_model = ChatHuggingFace(llm = llm)

model_template = tokenizer.chat_template
model_template

#chat_model._to_chat_prompt(msgs)

#res = chat_model.invoke(msgs)
#print(res.content)

from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Escreva um poema sobre {topic}")

prompt_template.invoke({"topic": "abacates"})

prompt_template

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente e está respondendo perguntas gerais."),
    ("user", "Explique-me em 1 parágrafo o conceito de {topic}")
])

prompt.invoke({"topic": "IA"})

chain = prompt | llm
chain.invoke({"topic": "IA"})

user_prompt = "Explique-me em 1 parágrafo o conceito de {topic}"
user_prompt_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>".format(user_prompt)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente e está respondendo perguntas gerais."),
    ("user", user_prompt_template)
])

chain = prompt | llm
chain.invoke({"topic": "IA"})

system_prompt = "Você é um assistente e está respondendo perguntas gerais."
user_prompt = "Explique para mim brevemente o conceito de {topic}, de forma clara e objetiva. Escreva em no máximo 1 parágrafo."

prompt = PromptTemplate.from_template(template.format(system_prompt = system_prompt, user_prompt = user_prompt))
prompt

prompt.invoke({"topic": "IA"})

user_prompt = "Explique para mim brevemente o conceito de {topic}, de forma clara e objetiva. Escreva em no máximo {tamanho}."
prompt = PromptTemplate.from_template(template.format(system_prompt=system_prompt, user_prompt=user_prompt))
prompt.invoke({"topic": "IA", "tamanho": "1 frase"})

prompt

chain = prompt | llm

resp = chain.invoke({"topic": "IA", "tamanho": "1 frase"})
resp

topic = "IA"  # @param {type:"string"}
tamanho = "1 parágrafo" # @param {type:"string"}

resp = chain.invoke({"topic": topic, "tamanho": tamanho})
resp

type(resp)

from langchain_core.output_parsers import StrOutputParser

chain_str = chain | StrOutputParser()

# Isso é equivalente a:
# chain_str = prompt | llm | StrOutputParser()

chain_str.invoke({"topic": "IA", "tamanho": "1 frase"})

len("teste teste teste".split())

from langchain_core.runnables import RunnableLambda
count = RunnableLambda(lambda x: f"Palavras: {len(x.split())}\n{x}")

chain = prompt | llm | StrOutputParser() | count
chain.invoke({"topic": "criptografia", "tamanho": "1 frase"})

for chunk in chain_str.stream({"topic": "buracos negros", "tamanho": "1 paragrafo"}):
  print(chunk, end="|")

from langchain_huggingface import HuggingFaceEndpoint

os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass()

#model_id = ""

llm_hub = HuggingFaceEndpoint(
    repo_id=model_id,
    temperature=0.1,
    max_new_tokens=512,
    model_kwargs={
        "max_length": 64,
    }
)

response = llm_hub.invoke("Quais os nomes dos planetas do sistema solar?")
print(response)

!pip install -qU langchain-openai

import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")

from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(model="gpt-4o-mini")

chatgpt = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

msgs = [
    (
        "system",
        "Você é um assistente prestativo que traduz do português para francês. Traduza a frase do usuário.",
    ),
    ("human", "Eu amo programação"),
]
ai_msg = chatgpt.invoke(msgs)
ai_msg

print(ai_msg.content)

"""
!pip install -q langchain-anthropic
import os
from getpass import getpass
from langchain_anthropic import ChatAnthropic

os.environ["ANTHROPIC_API_KEY"] = getpass("Anthropic API key: ")

model = ChatAnthropic(model='claude-3-opus-20240229', temperature=0.7)
res = model.invoke("Olá, como você está?")
print(res)"""

"""
!pip install -q langchain-google-genai
import os
from getpass import getpass
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = getpass("Google API key: ")

model = ChatGoogleGenerativeAI(model='gemini-pro')
res = model.invoke("Olá, como você está?")
print(res)"""
