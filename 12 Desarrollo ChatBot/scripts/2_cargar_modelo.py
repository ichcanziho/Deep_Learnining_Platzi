import os
from dotenv import load_dotenv
import openai

load_dotenv("../envs/ap.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="¿Quién descubrió América?",
    max_tokens=100
)

print(response.choices[0].text)
