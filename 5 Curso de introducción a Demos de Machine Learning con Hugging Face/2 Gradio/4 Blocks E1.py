import gradio as gr
from transformers import pipeline


def audio_a_text(audio):
    text = trans(audio)["text"]
    return text


def texto_a_sentimiento(text):
    sentiment = clasificador(text)[0]["label"]
    return sentiment


def make_block(dem):
    with dem:
        gr.Markdown("# Demo para la clase de Platzi")
        audio = gr.Audio(source="microphone", type="filepath")
        texto = gr.Textbox()
        b1 = gr.Button("Transcribe porfa")
        b1.click(audio_a_text, inputs=audio, outputs=texto)

        label = gr.Label()
        b2 = gr.Button("Clasifica porfa el sentimiento")
        b2.click(texto_a_sentimiento, inputs=texto, outputs=label)


if __name__ == '__main__':

    trans = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53-spanish")
    clasificador = pipeline("text-classification", model="pysentimiento/robertuito-sentiment-analysis")

    demo = gr.Blocks()
    make_block(demo)
    demo.launch()
