import os
from dotenv import load_dotenv
import requests
import time


def get_updates(token, offset=None):
    # definimos url
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    # asignamos params desde offset
    params = {"offset": offset} if offset else {}
    # obtenemos la respuesta http GET
    response = requests.get(url, params=params)
    # devolvemos en un JSON
    return response.json()


def print_new_messages(token):
    print("Iniciando Gabich_test_bot")
    # el siguiente por default no existe
    offset = None
    # Para que haga peticiones siempre
    while True:
        # obtenemos respuestas
        updates = get_updates(token, offset)
        # validamos que haya resultados desde http GET
        if "result" in updates:
            # imprimimos todas las respuestas
            for update in updates["result"]:
                message = update["message"]
                u_id = message["from"]["id"]
                username = message['from']["first_name"]
                text = message.get("text")
                print(f"Usuario: {username}({u_id})")
                print(f"Mensaje: {text}")
                print("-" * 20)
                # Pasar al siguiente
                offset = update["update_id"] + 1
        time.sleep(1)


if __name__ == '__main__':
    load_dotenv("../envs/ap.env")
    token_ = os.getenv("TELEGRAM_API_KEY")
    print_new_messages(token_)
