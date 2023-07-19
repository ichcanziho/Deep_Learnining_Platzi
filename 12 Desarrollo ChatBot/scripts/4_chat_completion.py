import os
from dotenv import load_dotenv
import openai

load_dotenv("../envs/ap.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "system", "content": "Eres un asistente que da informacion sobre deportes"},
        {"role": "user", "content": "¿Quién ganó el mundial de fútbol?"},
        {"role": "assistant", "content": "El mundial de 2022 lo ganó Argentina"},
        {"role": "user", "content": "¿Dónde se jugó?"}
    ],
    temperature=1,  # Este es el valor default
    max_tokens=60

)
print(response['choices'][0]['message']['content'])
print("*"*64)

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "system", "content": "Eres un asistente que da informacion sobre deportes"},
        {"role": "user", "content": "¿Quién ganó el mundial de fútbol?"},
        {"role": "assistant", "content": "El mundial de 2022 lo ganó Argentina"},
        {"role": "user", "content": "¿Dónde se jugó?"}
    ],
    temperature=0.2,
    max_tokens=60

)
print(response['choices'][0]['message']['content'])
print("*"*64)

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "system", "content": "Eres un asistente que da informacion sobre deportes."},
        {"role": "user", "content": "¿Quién ganó el mundial de fútbol?"},
        {"role": "assistant", "content": "El mundial de 2022 lo ganó Argentina"},
        {"role": "user", "content": "¿Dónde se jugó?"}
    ],
    temperature=1.9,
    max_tokens=60
)
print(response['choices'][0]['message']['content'])
