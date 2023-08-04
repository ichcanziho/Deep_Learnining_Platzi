import os
import requests
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Antecedentes 1: Cargar el API KEY de OpenAI como una variable de sistema.
from dotenv import load_dotenv
load_dotenv("../secret/keys.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Antecedentes 2: Instanciar dos LLMs de OpenAI un GPT3.5 y un Davinci
from langchain.llms import OpenAI
llm_gpt3_5 = OpenAI(
    model_name="gpt-3.5-turbo",
    n=1,
    temperature=0.3
)

llm_davinci = OpenAI(
    model_name="text-davinci-003",
    n=2,
    temperature=0.3
    )

# ----------------------------------------------------------------------

url = 'https://www.cs.virginia.edu/~evans/greatworks/diffie.pdf'
filename = "public_key_cryptography.pdf"

# Descargamos el archivo PDF del Url con un nombre `filename`
if not os.path.exists(filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f'Descargado {filename}')
else:
    print(f'{filename} ya existe, cargando desde el disco.')

# Convertimos el archivo PDF a un documento legible `Loader`
loader = PyPDFLoader(filename)
data = loader.load()

# Creamos nuestro generador de `embeddings`(vectorizador de texto) utilizando OpenAi como modelo base.
embeddings = OpenAIEmbeddings()
# Chroma nos permitirá convertir los datos de texto que hay en `data` en una base de datos vectorial utilizando a
# `embeddings" como método para conversión de texto a vector.
vectorstore = Chroma.from_documents(data, embeddings)

# Esto es la cantidad de paǵinas en el documento
print("páginas:", len(data))

from langchain.chains.summarize import load_summarize_chain

# La cadena load_summarize busca crear resumenes, pero también utilizamos `map_reduce` para que cada página sea
# resumida antes de hacer el resumen general del documento.
cadena_que_resume_documentos = load_summarize_chain(
    llm_davinci,
    chain_type="map_reduce"
)

ans = cadena_que_resume_documentos.run(data)
print("*"*64)
print(ans)

# Sin embargo, `map_reduce`no es la única chain type, vamos a probar ahora con `stuff` porque esta nos permite
# integrar `prompts` a nuestras peticiones. Adicionalmente, nuestro `prompt` será enviado a partir de un
# `Prompt template` para tener más control del mismo.

from langchain import PromptTemplate
plantilla = """Escribe un resumen bien chido del siguiente rollo:

{text}

RESUMEN CORTO CON SLANG MEXICANO:"""

# Dada la plantilla definimos que solo tenemos una variable de entrada `text`
prompt = PromptTemplate(
    template=plantilla,
    input_variables=["text"]
)

# Concatenamos nuestra plantilla al LLM davinci
cadena_que_resume_con_slang = load_summarize_chain(
    llm=llm_davinci,
    chain_type="stuff",
    prompt=prompt,
    verbose=True
)
# Vamos a observar la respuesta que nos da de resumen al utilizar solo las primeras 2 hojas de contenido
ans = cadena_que_resume_con_slang.run(data[:2])
print("*"*64)
print(ans)

# No siempre vamos a necesitar resúmenes del texto, que pasa si queremos responder alguna pregunta del texto?
from langchain.chains import RetrievalQA
# En ese sentido, NO vamos a usar un prompt template con el texto a analizar, sino que vamos a utilizar nuestra
# base de datos vectorial, y le vamos a pasar toda la información al `retriever`
cadena_que_resuelve_preguntas = RetrievalQA.from_chain_type(
    llm=llm_gpt3_5,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)
# Ahora ya podemos hacer prompts con preguntas a resolver.
ans = cadena_que_resuelve_preguntas.run("¿Cuál es la relevancia de la criptografía de llave pública?")
print("*"*64)
print(ans)