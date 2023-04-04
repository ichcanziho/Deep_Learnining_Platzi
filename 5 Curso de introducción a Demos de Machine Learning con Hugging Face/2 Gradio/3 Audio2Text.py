from transformers import pipeline
import gradio as gr


def transcribe(audio):
    text = modelo(audio)["text"]
    return text


if __name__ == '__main__':
    modelo = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53-spanish")

    gr.Interface(
        fn=transcribe,
        inputs=[gr.Audio(source="microphone", type="filepath")],
        outputs=["textbox"]
    ).launch()
