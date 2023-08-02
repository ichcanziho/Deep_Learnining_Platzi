from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv("../secret/keys.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm_gpt3_5 = OpenAI(
    model_name="gpt-3.5-turbo",
    n=1,
    temperature=0.3
)

prompt_argentino = """Respondé la pregunta basándote en el contexto de abajo. Si la
pregunta no puede ser respondida usando la información proporcionada,
respondé con "Ni idea, che".

Contexto: Los Modelos de Lenguaje de Gran Escala (MLGEs) son lo último en modelos usados en el Procesamiento del Lenguaje Natural (NLP).
Su desempeño superior a los modelos más chicos los hizo increíblemente
útiles para los desarrolladores que arman aplicaciones con NLP. Estos modelos
se pueden acceder vía la librería `transformers` de Hugging Face, vía OpenAI
usando la librería `openai`, y vía Cohere usando la librería `cohere`.

Pregunta: ¿Qué librerías están cerca de Buenos Aires?

Respuesta (escribe como argentina informal): """

print(llm_gpt3_5(prompt_argentino))
print("*"*64)

from langchain import PromptTemplate

plantilla_colombiana = """Responde a la pregunta con base en el siguiente contexto, parce. Si la
pregunta no puede ser respondida con la información proporcionada, responde
con "No tengo ni idea, ome".

Contexto: Los Modelos de Lenguaje Grandes (LLMs) son los últimos modelos utilizados en PNL.
Su rendimiento superior sobre los modelos más pequeños los ha hecho increíblemente
útiles para los desarrolladores que construyen aplicaciones habilitadas para PNL. Estos modelos
pueden ser accedidos a través de la biblioteca `transformers` de Hugging Face, a través de OpenAI
usando la biblioteca `openai`, y a través de Cohere usando la biblioteca `cohere`.

Pregunta: {pregunta}

Respuesta (escribe como colombiano informal): """

prompt_plantilla_colombiana = PromptTemplate(
    input_variables=["pregunta"],
    template=plantilla_colombiana
)


from langchain import LLMChain

llm_gpt3_5_chain = LLMChain(
    prompt=prompt_plantilla_colombiana,
    llm=llm_gpt3_5
)

pregunta = "Qué son los LLMs?"

ans = llm_gpt3_5_chain.run(pregunta)
print(ans)
print("*"*64)

pregunta = "Qué son las RAFGSERS?"

ans = llm_gpt3_5_chain.run(pregunta)
print(ans)
print("*"*64)
