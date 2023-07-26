from dotenv import load_dotenv
import requests
import openai
import time
import os


class ChatBotMaker:
    def __init__(self, env_file):
        load_dotenv(env_file)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.token = os.getenv("TELEGRAM_API_KEY")
        self.model_engine = os.getenv("MODEL_ENGINE")

    def get_updates(self, offset: int):
        """
        Función para obtener los mensajes más recientes del Bot de telegram
        :param offset: se utiliza para indicar el identificador del último mensaje recibido por el bot. Este parámetro
        se usa junto con el método "getUpdates" para obtener solo los mensajes nuevos que han llegado desde el último
        mensaje procesado por el bot.
        :return:
        """
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        params = {"timeout": 100, "offset": offset}
        response = requests.get(url, params=params)
        return response.json()["result"]

    def get_openai_response(self, prompt: str):
        """
        Genera una respuesta a un prompt de entrada utilizando el modelo de ChatGPT fine-tuned
        :param prompt: Mensaje de texto
        :return:
        """
        try:
            response = openai.Completion.create(
                engine=self.model_engine,
                prompt=prompt,
                max_tokens=200,
                n=1,
                temperature=0.5
            )
            return response.choices[0].text.strip()
        except openai.error.APIError as e:
            # Manejar error de API aquí, p. reintentar o iniciar sesión
            print(f"La API de OpenAI devolvió un error de API: {e}")
            pass  # Aprobar
        except openai.error.APIConnectionError as e:
            # Manejar error de conexión aquí
            print(f"Error al conectarse a la API de OpenAI: {e}")
            pass
        except openai.error.RateLimitError as e:
            # Manejar error de límite de tasa (recomendamos usar retroceso exponencial)
            print(f"La solicitud de API de OpenAI excedió el límite de frecuencia: {e}")
            pass

        return "Ocurrió un Error :("

    def send_messages(self, chat_id, text: str):
        """
        Envía un mensaje del BOT al Usuario de Telegram
        :param chat_id: Id del chat al cual será enviado el mensaje
        :param text: texto a enviar
        :return:
        """
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        params = {"chat_id": chat_id, "text": text}
        response = requests.post(url, params=params)
        return response



    def run(self):
        """
        Lógica para mantener corriendo el servicio de escucha de peticiones y generación de respuestas del ChatBot
        :return:
        """
        print("Starting bot...")
        offset = 0
        while True:
            # Escucha los nuevos mensajes
            updates = self.get_updates(offset)
            if updates:
                for update in updates:
                    offset = update["update_id"] + 1
                    chat_id = update["message"]["chat"]['id']
                    user_message = update["message"]["text"]
                    print(f"Received message: {user_message}")
                    # Genera una respuesta con ChatGPT
                    GPT = self.get_openai_response(user_message)
                    print(f"Answer generated: {GPT}")
                    # Regresa la respuesta al usuario de Telegram
                    self.send_messages(chat_id, GPT)
            else:
                time.sleep(1)
