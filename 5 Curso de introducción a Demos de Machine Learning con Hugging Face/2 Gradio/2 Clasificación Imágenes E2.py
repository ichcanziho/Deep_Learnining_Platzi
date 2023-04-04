import gradio as gr

titulo = "Mi primer demo con Hugging Face"
desc = "Este es un demo ejecutado durante la clase con Platzi."

gr.load(
    "huggingface/microsoft/swin-tiny-patch4-window7-224",
    inputs=gr.Image(label="Carga una imagen aquí"),
    title=titulo,
    description=desc
).launch()
