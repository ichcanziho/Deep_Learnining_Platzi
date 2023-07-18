from dotenv import load_dotenv
import random
import openai
import os


def get_base_clue():
    words = ['elefante', 'león', 'jirafa', 'hipopótamo', 'mono']
    random_word = random.choice(words)
    prompt = 'Adivina la palabra que estoy pensando. Es un animal que vive en la selva.'
    return prompt, random_word


def get_new_clue(animal):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt='Dame una caracteristica del tipo animal' + animal + ', pero jamás digas el nombre del animal',
        max_tokens=100)
    return response.choices[0].text


def play_game():
    # Empezamos con nuestro animal aleatorio y primer pista genérica
    first_clue, real_animal = get_base_clue()
    print(first_clue)
    # Mientras la respuesta del usuario sea diferente al verdadero animal
    while (user_input := input("Ingresa tu respuesta: ")) != real_animal:
        # Le decimos que se equivocó
        print('Respuesta incorrecta. Intentalo de nuevo')
        # Y le damos una nueva pista
        new_clue = get_new_clue(real_animal)
        print(new_clue)
    # Si salimos del ciclo while es porque el usuario ha acertado
    print('Correcto! La respuesta era:', real_animal)


if __name__ == '__main__':
    load_dotenv("../envs/ap.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    play_game()
