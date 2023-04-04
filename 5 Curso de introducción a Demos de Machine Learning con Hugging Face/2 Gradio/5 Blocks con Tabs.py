from transformers import pipeline
import gradio as gr
from PIL import Image


def classify_img(im):
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    ans = image_cla(im)
    labels = {v["label"]: v["score"] for v in ans}
    return labels


def voice2text(audio):
    text = voice_cla(audio)["text"]
    return text


def text2sentiment(text):
    sentiment = text_cla(text)[0]["label"]
    return sentiment


def make_block(dem):
    with dem:
        gr.Markdown("""

# Ejemplo de `space` multiclassifier:

Este `space` contiene los siguientes modelos:

- ASR: [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-large-xlsr-53-spanish)
- Text Classification: [Robertuito](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis)
- Image classifier: [Swin-small-patch4](https://huggingface.co/microsoft/swin-small-patch4-window7-224)

Autor del demo: [Gabriel Ichcanziho](https://www.linkedin.com/in/ichcanziho/)

Puedes probar un demo de cada uno en las siguientes pestañas:
        """)
        with gr.Tabs():
            with gr.TabItem("Transcribe audio en español"):
                with gr.Row():
                    audio = gr.Audio(source="microphone", type="filepath")
                    transcripcion = gr.Textbox()
                b1 = gr.Button("Voz a Texto")

            with gr.TabItem("Análisis de sentimiento en español"):
                with gr.Row():
                    texto = gr.Textbox()
                    label = gr.Label()
                b2 = gr.Button("Texto a Sentimiento")
            with gr.TabItem("Clasificación de Imágenes"):
                with gr.Row():
                    image = gr.Image(label="Carga una imagen aquí")
                    label_image = gr.Label(num_top_classes=5)
                b3 = gr.Button("Clasifica")

            b1.click(voice2text, inputs=audio, outputs=transcripcion)
            b2.click(text2sentiment, inputs=texto, outputs=label)
            b3.click(classify_img, inputs=image, outputs=label_image)


if __name__ == '__main__':
    image_cla = pipeline("image-classification", model="microsoft/swin-tiny-patch4-window7-224")
    voice_cla = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53-spanish")
    text_cla = pipeline("text-classification", model="pysentimiento/robertuito-sentiment-analysis")

    demo = gr.Blocks()
    make_block(demo)
    demo.launch()

