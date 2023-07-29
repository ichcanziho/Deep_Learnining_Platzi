import os
from dotenv import load_dotenv
from pprint import pprint

# leo el archivo keys.env y obtengo mi Api KEY de OpenAI
load_dotenv("../secret/keys.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

from langchain.llms import OpenAI

llm_gpt3_5 = OpenAI(
    model_name="gpt-3.5-turbo",
    n=1,
    temperature=0.3
)

print(llm_gpt3_5)
print("*"*64)
ans = llm_gpt3_5("Cómo puedo lograr una clase más interactiva para estudiantes virtuales?")
print(ans)
print("*"*64)

llm_davinci = OpenAI(
    model_name="text-davinci-003",
    n=2,
    temperature=0.3
    )

generacion = llm_davinci.generate(
    ["Dime un consejo de vida para alguien de 30 años", "Recomiendame libros similares a Hyperion Cantos"]
    )

pprint(generacion.generations)
print("*"*64)

pprint(generacion.llm_output)
print("*"*64)

n_tokens_preview = llm_gpt3_5.get_num_tokens("mis jefes se van a preocupar si gasto mucho en openai")
print(n_tokens_preview)
print("*"*64)
