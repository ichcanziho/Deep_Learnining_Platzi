import os
from dotenv import load_dotenv
load_dotenv("../secret/keys.env")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


from langchain.embeddings import OpenAIEmbeddings

embedding_openai = OpenAIEmbeddings(model="text-embedding-ada-002")

print(embedding_openai)

documentos_a_incrustar = [
    "Quiero que me expliques qué es el paracetamol",
    "¡Hola parce!",
    "¡Uy, hola!",
    "¿Cómo te llamas?",
    "Mis parceros me dicen Omar",
    "¡Hola Mundo!"
  ]

incrustaciones = embedding_openai.embed_documents(documentos_a_incrustar)

print(len(incrustaciones[3]))

consulta_incrustada = embedding_openai.embed_query(documentos_a_incrustar[0])

print(consulta_incrustada)
