#!/usr/bin/env python
# coding: utf-8

# # Projeto 01 - Transcrição e compreensão de vídeos
# 
# Neste projeto, vamos aprender a realizar transcrição e compreensão de vídeos. Ao final, você será capaz de criar sua própria aplicação que faz a sumarização automática de vídeos, permitindo que você entenda do que se trata e o que foi falado nele sem precisar assistí-lo.
# 
# Objetivos deste projeto:
# 
# * Compreender o conteúdo de um vídeo do Youtube sem precisar assisti-lo.
# * Pesquisar informações úteis no vídeo sem perder nenhum detalhe importante.
# * Interagir com o conteúdo do vídeo por meio de uma interface de chat (ou seja, como "conversar com o vídeo").
# 
# ## Instalação e Configuração
# 
# #!pip install -q langchain_core langchain_community langchain-huggingface langchain_ollama langchain_openai
# !pip install -q langchain_core==0.3.32 langchain_community==0.3.15 langchain-huggingface langchain_ollama langchain_openai
# 
# ### Instalação de bibliotecas para baixar transcrição
# 
# > **YouTube Transcript API**
# 
# Esta é uma API python que permite que você obtenha a transcrição/legendas para um determinado vídeo do YouTube. Ela também funciona para legendas geradas automaticamente e possui suporta a uma função que faz automaticamente a tradução de legendas
# 
# !pip install youtube-transcript-api==0.6.3
# 
# > **pytube**
# 
# Também é uma biblioteca que auxilia com o download de vídeos no youtube. Aqui ela não é necessária para baixar as transcrições dos vídeos, conseguimos ter acesso sem ela, mas iremos instalar também pois com ela podemos recuperar também demais informações do vídeo, como título, data de publicação, descrição, etc.
# 
# !pip install pytube
# 
# ## Importações
# 
# import os
# import io
# import getpass
# from langchain_community.document_loaders import YoutubeLoader
# from langchain_community.llms import HuggingFaceHub
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# 
# ## Carregando a transcrição
# 
# Para fazer o carregamento das transcrições usaremos o método YoutubeLoader(), que faz parte do dos document_loaders do LangChain. Veremos logo em seguida também como extrair os metadados do vídeo usando essa função
# 
# Através desses método conseguimos puxar as transcrições que já estão associadas ao vídeo e armazenadas no banco de dados do Youtube, o que irá nos economizar bastante processamento.
# 
# 
# O primeiro parâmetro é a URL do vídeo que queremos realizar a transcrição
# 
# Vamos pegar esse vídeo como primeiro exemplo https://www.youtube.com/watch?v=II28i__Tf3M
# 
# ### Defininido idiomas
# 
# O segundo parâmetro é o language. A função espera uma lista, nesse caso, uma lista de códigos de idioma em prioridade decrescente (por padrão).
# 
# Além de inglês ("en"), recomendamos deixar antes "pt" e "pt-BR" (ou "pt-PT") pois em alguns vídeos não possui "pt". Embora a grande maioria dos vídeos que testamos possua legenda com código "pt", mesmo para vídeos com a legenda em português brasileiro. Ou seja, deixamos assim pois em alguns vídeos do português brasileiro por exemplo o código é "pt", já para outros está como "pt-BR".
# 
# 
# video_loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=II28i__Tf3M",
#                                               language = ["pt", "pt-BR", "en"],)
# 
# Usaremos o .load() para fazer a leitura e ao mesmo tempo podemos passar as informações do vídeo para uma variável
# 
# infos = video_loader.load()
# infos
# 
# O valor de "page_content" corresponde à transcrição em si
# 
# para acessá-la devemos colocar `[0]` pois `infos` é uma lista, que nesse caso só tem o primeiro valor. O código então fica assim
# 
# transcricao = infos[0].page_content
# transcricao
# 
# Esse primeiro exemplo é de uma legenda que foi gerada automaticamente pelo sistema de reconhecimento de fala do youtube, que no geral tende a ser bom mas pode gerar erros, então não é perfeito. Mas ainda assim, dependendo da LLM ela vai entender que se trata de um erro com base no contexto
# 
# Para legendas automáticas verificamos que não houve perda considerável na compreensão, mas obviamento é esperado que uma legenda feita manualmente possua maiores chances de resultados melhores
# 
# ### Obter informações do vídeo
# 
# Note que carregamos a legenda/transcrição mas nenhuma outra informação sobre o vídeo, o que pode ser útil depedendo do nosso objetivo.
# 
# > Há duas maneiras de obter informações do vídeo:
# 
# 1. Usar o parâmetro `add_video_info` na função YoutubeLoader.from_youtube_url(). Por trás dos panos, a função se comunica com a biblioteca pytube. No entanto, há um bug conhecido com o pytube que pode ocorrer ocasionalmente e que até o momento não foi definitivamente resolvido. Obs: Esse bug já existe há algum tempo, sem um prazo claro para uma correção completa pelos autores. É por isso que mostraremos um segundo método.
# 
# 2. Usar a biblioteca `BeautifulSoup` (ou similar). Embora essa abordagem seja um pouco mais manual, mostramos como ela ainda pode ser feita facilmente com apenas algumas linhas de código.
# 
# 
# 
# #### método 1: Com parâmetro add_video_info
# 
# Podemos passar como parâmetro add_video_info=True (que por padrão é =False) e isso fará com que sejam retornados os metadados do vídeo, como: título, descrição, autor, visualizações, e capa)
# 
# Para usar esse parâmetro você precisa ter instalado antes a biblioteca pytube (lembre-se que instalamos anteriormente neste Colab, então se estiver executando em seu computador certifique-se que instalou essa biblioteca)
# 
# video_loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=II28i__Tf3M",
#                                                #add_video_info = True,
#                                                language = ["pt", "pt-BR", "en"],)
# 
# infos = video_loader.load()
# 
# infos
# 
# Podemos organizar desse modo
# 
# infos_video = f"""Informações do vídeo:
# 
# Título: {infos[0].metadata['title']}
# Autor: {infos[0].metadata['author']}
# Data: {infos[0].metadata['publish_date'][:10]}
# URL: https://www.youtube.com/watch?v={infos[0].metadata['source']}
# 
# Transcrição: {transcricao}
# """
# print(infos_video)
# 
# > **Observação importante:** Se aparecer esse erro para você "PytubeError: Exception while accessing title of..." tente reiniciar a sessão do Colab e executar novamente o mesmo código na ordem. Essa mensagem é referente a um bug da biblioteca usada para ler os dados do Youtube e que até o momento não há um padrão conhecido ou uma explicação dos autores (se esse bug for definitivamente resolvido iremos remover esse aviso). Se esse erro continuar a aparecer para você, use o método 2 abaixo
# 
# #### método 2: Com BeautifulSoup
# 
# Para fazer desse modo, precisaremos importar a biblioteca `requests` e o método `bs4` do `beautifulsoup`, que já está instalado por padrão no Colab. Então, ao invés do `infos = video_loader.load()` (conforme explicado no método 1 acima) você usará esse código abaixo
# * explicação rápida sobre o código: é feita a busca no conteúdo HTML da página do vídeo no YouTube usando a função `requests.get()` e, em seguida, analisa-o com o BeautifulSoup. Ele procura a tag usando `soup.find_all()`, que retorna uma lista de tags correspondentes. O título do vídeo é extraído pegando o primeiro elemento, convertendo-o em uma string e, em seguida, removendo o `<title>` e as tags com `replace()`. Ao final, o código imprime o título "limpo".
# 
# import requests
# from bs4 import BeautifulSoup
# 
# def get_video_title(url):
#   r = requests.get(url)
#   soup = BeautifulSoup(r.text)
# 
#   link = soup.find_all(name="title")[0]
#   title = str(link)
#   title = title.replace("<title>","")
#   title = title.replace("</title>","")
# 
#   return title
# 
# video_url = "https://www.youtube.com/watch?v=II28i__Tf3M"
# 
# video_title = get_video_title(video_url)
# video_title
# 
# ### Reunindo as informações
# 
# Agora, vamos apenas combinar as informações do vídeo com a transcrição que obtivemos anteriormente. Salvaremos tudo em uma mesma variável chamada `infos_video`
# 
# infos_video = f"""Informações do Vídeo:
# 
# Título: {video_title}
# URL: {video_url}
# 
# Transcrição: {transcricao}
# """
# print(infos_video)
# 
# ## Salvando transcrição em um arquivo
# 
# Esse código abre um arquivo chamado "transcricao.txt" em modo de escrita ("w") com codificação UTF-8;  dentro do bloco `with` ele grava dados no arquivo.
# 
# Para cada item na variável `infos`, ele escreve o conteúdo da variável `infos_video` no arquivo. O uso do bloco with garante que o arquivo seja fechado corretamente após a gravação, mesmo que ocorra algum erro durante a execução. E aqui não precisa do `f.close()` pois o bloco with fecha o arquivo automaticamente ao finalizar
# 
# with io.open("transcricao.txt", "w", encoding="utf-8") as f:
#   for doc in infos:
#     f.write(infos_video)
# 
# ## Carregamento do modelo
# 
# Vamos reaproveitar as funções de carregamento que usamos nos projeos anteriores, basta copiar e colar
# 
# from langchain_huggingface import HuggingFaceEndpoint
# 
# def model_hf_hub(model = "meta-llama/Meta-Llama-3-8B-Instruct", temperature = 0.1):
#   llm = HuggingFaceEndpoint(repo_id = model,
#                        temperature = temperature,
#                        return_full_text = False,
#                        max_new_tokens = 1024,
#                        task="text-generation"
#                        )
#   return llm
# 
# def model_openai(model = "gpt-4o-mini", temperature = 0.1):
#   llm = ChatOpenAI(model = model, temperature = temperature)
#   return llm
# 
# def model_ollama(model = "phi3", temperature = 0.1):
#   llm = ChatOllama(model = model, temperature = temperature)
#   return llm
# 
# E aqui no Colab precisar setar as variáveis de ambiente. Pode usar o .env também, especialmente se estiver executando localmente
# 
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass()
# os.environ["HF_TOKEN"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]
# 
# os.environ["OPENAI_API_KEY"] = getpass.getpass()
# 
# model_class = "hf_hub" # @param ["hf_hub", "openai", "ollama"]
# 
# if model_class == "hf_hub":
#   llm = model_hf_hub()
# elif model_class == "openai":
#   llm = model_openai
# elif model_class == "ollama":
#   llm = model_ollama
# 
# 
# model_class, llm
# 
# ## Criação do prompt template
# 
# Vamos manter a base do prompt simples, basicamente instruindo que deve responder com base na transcrição fornecida. Você pode modificá-lo à vontade depois, para deixar mais adequado ao seu objetivo ou simplesmente para tentar alcançar melhores resultados
# 
# * Aqui vamos passar o transcrição completa. Estaremos lidando com modelos que possuem uma janela grande de contexto - por exemplo o llama 3 possui algo em torno de 8k, já o chatGPT 4o por exemplo possui ainda mais. Deve ser uma capacidade suficiente de leitura de tokens de entrada para lidar com a maioria das transcrições dos vídeos, e será para todos os testados aqui.
# * Como a ideia desse projeto é criar uma ferramenta que faz o resumo / sumarização então adicionar a transcrição inteira como contexto é até uma opção mais interessante, já que para RAG é recuperado geralmente uma quantidade limite de pedaços de documento. Portanto, se fosse usado RAG teria que configurar bem os parâmetros, provavelmente escolher um valor maior de k por exemplo para recuperar mais documentos (no entanto, lembre-se que elevar muito esse valor aumenta o custo computacional da aplicação)
# * Mas caso o vídeo seja realmente grande então pode ser interessante dividir em partes. Para isso sugerimos usar o código do projeto 3, pode copiar as funções prontas que fazem as etapas de indexação e recuperação (indexing & retrieval)
# 
# Além da transcrição, o prompt template irá aceitar a variável consulta, que nada mais é do que a entrada para a LLM, que pode ser uma pergunta ou instrução
# 
# E o `if model_class.startswith("hf"):` apenas copiamos do projeto anterior, lembrando que isso é para melhorar os resultados com a implementação via Hugging Face Hub, que até o momento funciona melhor se especificarmos manualmente os tokens de início e fim. Aqui o template é do llama 3.x, mas for usar outro modelo open source que exija template diferente então lembre de mudar.
# 
# 
# system_prompt = "Você é um assistente virtual prestativo e deve responder a uma consulta com base na transcrição de um vídeo, que será fornecida abaixo."
# 
# inputs = "Consulta: {consulta} \n Transcrição: {transcricao}"
# 
# if model_class.startswith("hf"):
#   user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>".format(inputs)
# else:
#   user_prompt = "{}".format(inputs)
# 
# prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
# 
# prompt_template
# 
# ## Criação da chain
# 
# Nossa chain ficará assim
# 
# chain = prompt_template | llm | StrOutputParser()
# 
# ## Geração da resposta
# 
# Por fim, vamos gerar o resultado, fornecendo como parâmetro a transcrição e a consulta que queremos (podendo ser pergunta, instrução, etc.)
# 
# res = chain.invoke({"transcricao": transcricao, "consulta": "resuma"})
# print(res)
# 
# Podemos melhorar esse prompt (consulta), deixando algo como `"sumarize de forma clara de entender`
# 
# res = chain.invoke({"transcricao": transcricao, "consulta": "sumarize de forma clara de entender"})
# print(res)
# 
# res = chain.invoke({"transcricao": transcricao, "consulta": "explique em 1 frase sobre o que fala esse vídeo"})
# print(res)
# 
# res = chain.invoke({"transcricao": transcricao, "consulta": "liste os temas desse video"})
# print(res)
# 
# ## Tradução da transcrição
# 
# Para os modelos mais modernos não é necessário traduzir antes, pode carregar a transcrição no idioma desejado e passar para a LLM mesmo que no idioma diferente daquele que você escreveu as instruções no prompt template, isso porque o modelo deve ser capaz de entender.
# 
# Mas também é possível traduzir a transcrição usando essa mesma ferramenta.
# Isso pode ser muito útil caso o modelo que esteja trabalhando não funcione bem para múltiplos idiomas.
# 
# Para implementar isso, basta definirmos para o parâmetro translation o código do idioma para o qual desejamos traduzir.
# Por exemplo para o francês ficaria assim
# 
# url_video = "https://www.youtube.com/watch?v=II28i__Tf3M"
# 
# video_loader = YoutubeLoader.from_youtube_url(
#     url_video,
#     add_video_info=True,
#     language=["pt", "en"],
#     translation="fr",
# )
# 
# infos = video_loader.load()
# transcricao = infos[0].page_content
# transcricao
# 
# ## Junção da pipeline em funções
# 
# Para deixar mais prático e evitar repetições do código vamos reunir toda a nossa lógica em funções, assim não vai ser mais necessário ficar copiando e colando o código toda vez que for testar em outro vídeo
# 
# def llm_chain(model_class):
#   system_prompt = "Você é um assistente virtual prestativo e deve responder a uma consulta com base na transcrição de um vídeo, que será fornecida abaixo."
# 
#   inputs = "Consulta: {consulta} \n Transcrição: {transcricao}"
# 
#   if model_class.startswith("hf"):
#       user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>".format(inputs)
#   else:
#       user_prompt = "{}".format(inputs)
# 
#   prompt_template = ChatPromptTemplate.from_messages([
#       ("system", system_prompt),
#       ("user", user_prompt)
#   ])
# 
#   ### Carregamento da LLM
#   if model_class == "hf_hub":
#       llm = model_hf_hub()
#   elif model_class == "openai":
#       llm = model_openai()
#   elif model_class == "ollama":
#       llm = model_ollama()
# 
#   chain = prompt_template | llm | StrOutputParser()
# 
#   return chain
# 
# def get_video_info(url_video, language="pt", translation=None):
# 
#   video_loader = YoutubeLoader.from_youtube_url(
#       url_video,
#       #add_video_info=False,
#       language=language,
#       translation=translation,
#   )
# 
#   infos = video_loader.load()[0]
#   transcript = infos.page_content
#   video_title = get_video_title(url_video)
# 
#   return transcript, metadata
# 
# Vamos testar abaixo
# 
# transcript, metadata = get_video_info("https://www.youtube.com/watch?v=II28i__Tf3M")
# 
# metadata, transcript
# 
# Aqui aproveitamos para adicionar um Tratamento de erro com Try Catch, pois caso não haja uma transcrição para esse vídeo será retornado um erro (possivelmente será esse: `IndexError: list index out of range`), com isso não será possível fazer os processamentos seguintes. Por isso programos aqui para o programa parar interromper a execuão se esse for caso
# 
# def interpret_video(url, query="resuma", model_class="hf_hub", language="pt", translation=None):
# 
#   try:
#     transcript, metadata = get_video_info(url, language, translation)
# 
#     chain = llm_chain(model_class)
# 
#     res = chain.invoke({"transcricao": transcript, "consulta": query})
#     print(res)
# 
#   except Exception as e:
#     print("Erro ao carregar transcrição")
#     print(e)
# 
# ## Geração final
# 
# Podemos definir uma interface mais apresentável no Colab através dos comandos para deixar as variáveis em formato de valores de formulário.
# 
# Ideias do que adicionar à query:
# 
# * `sumarize de forma clara de entender`
# * `liste os temas desse vídeo`
# * `explique em 1 frase sobre o que fala esse vídeo`
# 
# url_video = "https://www.youtube.com/watch?v=II28i__Tf3M" # @param {type:"string"}
# query_user = "sumarize de forma clara de entender" # @param {type:"string"}
# model_class = "openai" # @param ["hf_hub", "openai", "ollama"]
# language = ["pt", "pt-BR", "en"] # @param {type:"string"}
# 
# interpret_video(url_video, query_user, model_class, language)
# 
# url_video = "https://www.youtube.com/watch?v=rEE8ERGKsqo" # @param {type:"string"}
# query_user = "sumarize de forma clara de entender" # @param {type:"string"}
# model_class = "hf_hub" # @param ["hf_hub", "openai", "ollama"]
# language = ["pt", "pt-BR", "en"] # @param {type:"string"}
# 
# interpret_video(url_video, query_user, model_class, language)
# 
# ## Explorando mais
# 
# Vamos deixar nossa aplicação mais interessante. Podemos fazer mais de uma consulta/query de uma vez por vídeo.
# 
# * Para cada vídeo informado, podemos definir para exibir informações como o titulo dele;
# * e logo abaixo um resumo em um paragrafo;
# * depois uma lista de temas abordados, etc.
# 
# No Colab temos um modo interessante de fazer isso, que é através do Markdown.
# 
# 
# ### Exemplo com Markdown
# 
# Markdown é um formato de linguagem de marcação muito adotada pela comunidade por ser simples e leve, permitindo criar texto formatado com uso de símbolos especiais, como asteriscos e colchetes, em vez de tags de HTML. No Google Colab, o uso de markdown pode tornar a visualização do texto mais interessante e fácil de ler, facilitando a compreensão e a apresentação de informações. Por exemplo, usando markdown, você pode deixar um texto em *itálico* ao deixar dentro de asteriscos, ou **negrito** se deixar ele dentro de asteriscos duplos. Também podemos adicionar títulos com diferentes níveis, por exemplo para nível 1,2,3 basta colocar antes da frase `#` `##` ou `###` respectivamente
# 
# > Mais sobre a sintaxe aqui: https://www.markdownguide.org/basic-syntax/
# 
# texto = """
# ### Título
# descrição em **destaque** *aqui...*
# """
# 
# from IPython.display import Markdown
# 
# display(Markdown(texto))
# 
# E abaixo em forma de lista por exemplo
# 
# Com isso, se pedirmos por exemplo pra LLM retornar uma lista então ficará mais apresentável também, já que ela irá retornar nesse formato com * no inicio de cada item, já que é um padrão comum
# 
# Nos projetos em que usamos os Streamlit talvez tenha notada que já ficou automaticamente dessem modo, pois a interface já interpreta corretamente markdown
# 
# lista = """
# * item
# * item
# * item
# """
# 
# display(Markdown(lista))
# 
# ### Finalização do código
# 
# Copiamos do código que fizemos antes, só mudando de infos[0].metadata para metadata
# 
# e ao invés do print() normal, vamos colocar no lugar display(Markdown()) para deixar mais apresentável
# 
# Como resposta, teremos:
# * Informações do vídeo - que são os metadados: título, autor, data, URL.
# 
# *  Sobre o que fala o vídeo - resumo de 1 frase, para contextualizarmos bem rapidamente
# 
# * Temas - listagem dos principais temas desse vídeo
# 
# * Resposta para a consulta - que é a resposta para a consulta personalizada que fizemos, que pode ser uma pergunta ou instrução
# 
# Aqui você pode customizar à vontade depois, deixar com as consultas que achar mais conveniente para o objetivo de sua aplicação
# 
# Você poderia também separar em duas funções: uma para exibir junto as consultas fixas (informações do vídeo, resumo e temas) e outra para exibir a consulta personalizada, podendo deixar em formato de chat, igual feito no projeto 1 e 2.
# 
# mas aqui estamos deixando mais simples e rápido, portanto ficou desse modo
# 
# def interpret_video(url, query="liste os temas desse vídeo", model_class="hf_hub", language="pt", translation=None):
# 
#   try:
#     transcript, video_title = get_video_info(url, language, translation)
# 
#     infos_video = f"""## Informações do vídeo
# 
#     Título: {video_title}
#     URL: {url}
# 
#     """
# 
#     display(Markdown(infos_video))
# 
#     chain = llm_chain(model_class)
# 
#     t = "\n## Sobre o que fala o vídeo \n"
#     res = chain.invoke({"transcricao": transcript, "consulta": "explique em 1 frase sobre o que fala esse vídeo. responda direto com a frase"})
#     display(Markdown(t + res))
# 
#     t = "\n## Temas \n"
#     res = chain.invoke({"transcricao": transcript, "consulta": "lista os principais temas desse vídeo"})
#     display(Markdown(t + res))
# 
#     t = "\n## Resposta para a consulta \n"
#     res = chain.invoke({"transcricao": transcript, "consulta": query})
#     display(Markdown(t + res))
# 
#   except Exception as e:
#     print("Erro ao carregar transcrição")
#     print(e)
# 
# url_video = "https://www.youtube.com/watch?v=rEE8ERGKsqo" # @param {type:"string"}
# query_user = "sumarize de forma clara de entender" # @param {type:"string"}
# model_class = "hf_hub" # @param ["hf_hub", "openai", "ollama"]
# language = ["pt", "pt-BR", "en"] # @param {type:"string"}
# 
# interpret_video(url_video, query_user, model_class, language)
# 
# url_video = "https://www.youtube.com/watch?v=n9u-TITxwoM" # @param {type:"string"}
# query_user = "sumarize de forma clara de entender" # @param {type:"string"}
# model_class = "hf_hub" # @param ["hf_hub", "openai", "ollama"]
# language = ["pt", "pt-BR", "en"] # @param {type:"string"}
# 
# interpret_video(url_video, query_user, model_class, language)
# 
# url_video = "https://www.youtube.com/watch?v=XXcHDy3QH-E" # @param {type:"string"}
# query_user = "sumarize de forma clara de entender" # @param {type:"string"}
# model_class = "hf_hub" # @param ["hf_hub", "openai", "ollama"]
# language = ["pt", "pt-BR", "en"] # @param {type:"string"}
# 
# interpret_video(url_video, query_user, model_class, language)
# 
# ## Reconhecimento de fala
# 
# Se o vídeo não tiver uma transcrição disponível, será necessário gerar uma de forma automática utilizando um modelo de reconhecimento de fala, também conhecido como Speech-to-Text (STT). Esses modelos convertem fala em texto a partir de arquivos de áudio, permitindo que o conteúdo do vídeo seja transcrito para ser processado pela nossa aplicação de sumarização.
# 
# Um exemplo popular de modelo de Speech-to-Text é o [Whisper](https://openai.com/index/whisper/), da OpenAI, que oferece uma solução robusta para transcrição automática. Os preços e detalhes sobre o uso desse modelo podem ser encontrados na página de "[pricing](https://openai.com/api/pricing/)" da OpenAI. No entanto, neste projeto, optamos por não utilizá-lo, pois todos os vídeos testados já possuíam transcrições, muitas vezes geradas automaticamente pelo YouTube. Essa abordagem economiza processamento, já que o foco principal foi a implementação de modelos de linguagem grandes (LLMs) e a exploração de ferramentas do Langchain.
# 
# * Se você precisar integrar um serviço de transcrição, a implementação é simples, podendo ser feita com uma chamada de função. A documentação do Langchain oferece instruções detalhadas sobre como realizar essa integração aqui: https://js.langchain.com/v0.2/docs/integrations/document_loaders/file_loaders/openai_whisper_audio/.
# 
# * Na verdade você tem a liberdade de escolher qualquer serviço ou código para realizar o reconhecimento de fala. Mesmo que o serviço não se integre diretamente ao Langchain, isso não é um problema, pois o essencial é obter o texto final. No final, o que você obtém é um texto normal, que pode ser armazenado em um arquivo de texto ou diretamente em uma variável.
# 
# * Entretanto, se desejar aproveitar soluções prontas dentro do Langchain, você pode utilizar o Whisper da OpenAI ou outras opções como o [GoogleSpeechToText](https://python.langchain.com/v0.2/docs/integrations/document_loaders/google_speech_to_text/) e o [AssemblyAI Audio Transcripts](https://python.langchain.com/v0.2/docs/integrations/document_loaders/assemblyai/), que também são altamente eficazes e fáceis de integrar ao langchain.
# 
# ## Indo além 🚀
# 
# 1. Conforme mencionado anteriormente, para vídeos muito longos ou transcrições maiores, pode ser necessário dividir a transcrição em partes menores e aplicar técnicas de indexação e recuperação utilizando RAG.
#  * Isso é essencial caso o vídeo que deseja processar possua horas por exemplo - talvez modelos muito modernos e com janela de contexto maior consigam lidar, mas mesmo que consigam pode ser interessante aplicar essas técnicas, principalmente caso o que busque não seja a sumarização mas sim algo próximo do nosso projeto de perguntas e respostas com documentos, que será visto no projeto 3.
#  * Caso deseje seguir esse caminho, basta reutilizar o código já pronto no projeto 3, que lida com esse tipo de segmentação e recuperação de forma eficiente. Assim, você poderá manter a integridade da transcrição, otimizando o processamento e a geração de respostas mais precisas, mesmo com grandes volumes de dados.
# 
# 2. Como ideia adicional ou desafio: é possível personalizar este projeto para atender a outras necessidades, ou ainda usar a sumarização como uma etapa posterior dentro de uma pipeline maior.
#   * Um exemplo seria o uso da ferramenta [YouTubeSearchTool](https://python.langchain.com/v0.2/docs/integrations/tools/youtube//) do Langchain, que permite buscar automaticamente uma lista de URLs de vídeos no YouTube com base em um tema ou palavra-chave fornecida pelo usuário.
#   * Nessa abordagem, você poderia implementar uma aplicação que solicita um termo de busca e, em seguida, utiliza essa ferramenta para buscar vídeos relevantes, retornando em uma lista. Por fim, basta criar um laço de repetição que execute a função interpret_video (que criamos nesse projeto) para cada vídeo da lista, realizando assim a sumarização de múltiplos vídeos associados a um tema de forma automatizada.
# 
# 3. Além disso, é possível integrar essa funcionalidade a uma interface interativa com ferramentas como Streamlit, que permite criar facilmente interfaces gráficas.
#  * Os métodos para essa integração estão descritos detalhadamente nos projetos 2 e 3, caso queira expandir a aplicação.
#  * Embora, neste exemplo, tenhamos optado por usar o Google Colab pela conveniência e para demonstrar a exibição com Markdown, a migração para uma interface mais robusta pode ser feita com facilidade e oferecer uma experiência mais fluida para o usuário final caso deseje publicar essa aplicação.
# 
