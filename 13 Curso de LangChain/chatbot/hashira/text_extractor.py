import datetime
import json
import os
import re
from typing import Dict

import emoji
import requests
from termcolor import colored
from utils import create_dir, load_config, remove_existing_file
from dotenv import load_dotenv


def preprocess_text(text: str) -> str:
    """
    Preprocesa el texto eliminando ciertos patrones y caracteres.

    Args:
        text (str): Texto a preprocesar.

    Returns:
        El texto preprocesado.
    """
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"Copyright.*", "", text)
    text = text.replace("\n", " ")
    text = emoji.demojize(text)
    text = re.sub(r":[a-z_&+-]+:", "", text)
    return text


def download_file(url: str, repo_info: dict, jsonl_file_name: str) -> None:
    """
    Descarga un archivo desde una URL y lo guarda en un archivo JSONL.

    Args:
        url (str): URL desde donde se descarga el archivo.
        repo_info (dict): Información sobre el repositorio desde donde se descarga el archivo.
        jsonl_file_name (str): Nombre del archivo JSONL donde se guarda el archivo descargado.
    """
    response = requests.get(url)
    filename = url.split("/")[-1]
    text = response.text

    if text is not None and isinstance(text, str):
        text = preprocess_text(text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        file_dict = {
            "title": filename,
            "repo_owner": repo_info["owner"],
            "repo_name": repo_info["repo"],
            "text": text,
        }

        with open(jsonl_file_name, "a") as jsonl_file:
            jsonl_file.write(json.dumps(file_dict) + "\n")
    else:
        print(f"Texto no esperado: {text}")


def process_directory(
    path: str,
    repo_info: Dict,
    headers: Dict,
    jsonl_file_name: str,
) -> None:
    """
    Procesa un directorio de un repositorio de GitHub y descarga los archivos en él.

    Args:
        path (str): Ruta del directorio a procesar.
        repo_info (Dict): Información sobre el repositorio que contiene el directorio.
        headers (Dict): Headers para la petición a la API de GitHub.
        jsonl_file_name (str): Nombre del archivo JSONL donde se guardarán los archivos descargados.
    """
    # Si el nombre del directorio es 'zh', lo omite y retorna inmediatamente.
    # Esta característica está implementada para no descargar las traducciones en chino.
    if os.path.basename(path) == "zh":
        print(
            colored(
                f"Se omite el directorio 'zh' (traducciones en chino): {path}", "yellow"
            )
        )
        return

    base_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['repo']}/contents/"
    print(
        colored(f"Procesando directorio: {path} del repo: {repo_info['repo']}", "blue")
    )
    response = requests.get(base_url + path, headers=headers)

    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file["type"] == "file" and (
                file["name"].endswith(".mdx") or file["name"].endswith(".md")
            ):
                print(colored(f"Descargando documento: {file['name']}", "green"))
                print(colored(f"Descarga URL: {file['download_url']}", "cyan"))
                download_file(
                    file["download_url"],
                    repo_info,
                    jsonl_file_name,
                )
            elif file["type"] == "dir":
                process_directory(
                    file["path"],
                    repo_info,
                    headers,
                    jsonl_file_name,
                )
        print(colored("Exito en extracción de documentos del directorio.", "green"))
    else:
        print(
            colored(
                "No se pudieron recuperar los archivos. Verifique su token de GitHub y los detalles del repositorio.",
                "red",
            )
        )


def main():
    """
    Función principal que se ejecuta cuando se inicia el script.
    """
    config = load_config()
    load_dotenv("../.env")
    github_token = os.getenv("GITHUB_TOKEN")
    print(github_token)
    os.environ['OPENAI_API_KEY'] = github_token

    if github_token is None:
        raise ValueError(
            "GITHUB_TOKEN no está configurado en las variables de entorno."
        )
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3.raw",
    }

    current_date = datetime.date.today().strftime("%Y_%m_%d")
    jsonl_file_name = f"data/docs_en_{current_date}.jsonl"

    create_dir("data/")
    remove_existing_file(jsonl_file_name)

    for repo_info in config["github"]["repos"]:
        process_directory(
            repo_info["path"],
            repo_info,
            headers,
            jsonl_file_name,
        )


if __name__ == "__main__":
    main()
