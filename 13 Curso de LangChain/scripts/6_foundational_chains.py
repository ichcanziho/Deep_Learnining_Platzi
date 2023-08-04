# Antecedentes 1: Cargar el API KEY de OpenAI como una variable de sistema.
import os
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

import re


def limpiar_texto(entradas: dict) -> dict:
    texto = entradas["texto"]

    # Eliminamos los emojis utilizando un amplio rango unicode
    # Ten en cuenta que esto podría potencialmente eliminar algunos caracteres válidos que no son en inglés
    patron_emoji = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticonos
        "\U0001F300-\U0001F5FF"  # símbolos y pictogramas
        "\U0001F680-\U0001F6FF"  # símbolos de transporte y mapas
        "\U0001F1E0-\U0001F1FF"  # banderas (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE,
    )
    texto = patron_emoji.sub(r'', texto)

    # Removemos las URLs
    patron_url = re.compile(r'https?://\S+|www\.\S+')
    texto = patron_url.sub(r'', texto)

    return {"texto_limpio": texto}


from langchain.chains import TransformChain

cadena_que_limpia = TransformChain(
    input_variables=["texto"],
    output_variables=["texto_limpio"],
    transform=limpiar_texto
)

clean = cadena_que_limpia.run('Chequen está página https://twitter.com/home 🙈')
print(clean)

from langchain import PromptTemplate
from langchain.chains import LLMChain

# Empezamos creando nuestro prompt template que recibe como parámetro un 'texto_limpio' (salida de de la cadena de limpieza)
# y lo parafrasea con un estilo informa de una persona (estilo).
plantilla_parafrasea = """Parafrasea este texto:

{texto_limpio}

En el estilo de una persona informal de {estilo}.

Parafraseado: """

# Dado que nuestro Template tiene 2 variables, debemos indicarlas en el parámetro `input_variables
prompt_parafraseo = PromptTemplate(
    input_variables=["texto_limpio", "estilo"],
    template=plantilla_parafrasea
)

# Ahora solo falta crear la cadena que cambia estilo utilizando como LLM a GPT3.5, esta cadena terminará creando una variable
# a la salida llamada `texto_final`
cadena_que_cambia_estilo = LLMChain(
    llm=llm_gpt3_5,
    prompt=prompt_parafraseo,
    output_key='texto_final'
)

# Texto_final es la variable de entrada, puesto que así la definimos en la cadena de parafraseo
plantilla_resumen = """Resume este texto:

{texto_final}

Resumen: """

prompt_resumen = PromptTemplate(
    input_variables=["texto_final"],
    template=plantilla_resumen
)

# Texto resumido será la variable final con la que termina nuestra secuencia de cadenas
cadena_que_resume = LLMChain(
    llm=llm_gpt3_5,
    prompt=prompt_resumen,
    output_key="texto_resumido"
)

from langchain.chains import SequentialChain

cadena_secuencial = SequentialChain(
    chains=[cadena_que_limpia, cadena_que_cambia_estilo, cadena_que_resume],
    input_variables=["texto", "estilo"],
    output_variables=["texto_resumido"]
)

texto_entrada = """
¡Monterrey es una ciudad impresionante! 🏙️
Es conocida por su impresionante paisaje de montañas ⛰️ y su vibrante cultura norteña.
¡No olvides visitar el famoso Museo de Arte Contemporáneo (MARCO)!
🖼️ Si eres fanático del fútbol, no puedes perderte un partido de los Rayados o de los Tigres. ⚽
Aquí te dejo algunos enlaces para que puedas conocer más sobre esta maravillosa ciudad:
https://visitamonterrey.com, https://museomarco.org, https://rayados.com, https://www.tigres.com.mx.
¡Monterrey te espera con los brazos abiertos! 😃🇲🇽

Monterrey es la capital y ciudad más poblada del estado mexicano de Nuevo León, además de la cabecera del 
municipio del mismo nombre. Se encuentra en las faldas de la Sierra Madre Oriental en la región noreste de 
México. La ciudad cuenta según datos del XIV Censo de Población y Vivienda del Instituto Nacional de 
Estadística y Geografía de México (INEGI) en 2020 con una población de 3 142 952 habitantes, por lo cual 
de manera individual es la 9.ª ciudad más poblada de México, mientras que la zona metropolitana de Monterrey 
cuenta con una población de 5 341 175 habitantes, la cual la convierte en la 2.ª área metropolitana más 
poblada de México, solo detrás de la Ciudad de México.8​

La ciudad fue fundada el 20 de septiembre de 1596 por Diego de Montemayor y nombrada así en honor al castillo 
de Monterrey en España. Considerada hoy en día una ciudad global, es el segundo centro de negocios y finanzas 
del país, así como una de sus ciudades más desarrolladas, cosmopolitas y competitivas. Sirve como el 
epicentro industrial, comercial y económico para el Norte de México.9​ Según un estudio de Mercer Human 
Resource Consulting, en 2019, fue la ciudad con mejor calidad de vida en México y la 113.ª en el mundo.10​ 
La ciudad de Monterrey alberga en su zona metropolitana la ciudad de San Pedro Garza García, la cual es el 
área con más riqueza en México y América Latina.11​
"""

print("*" * 64)
ans = cadena_secuencial({'texto': texto_entrada, 'estilo': 'ciudad de méxico'})
print(ans)
