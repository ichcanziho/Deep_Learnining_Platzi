import gradio as gr


def saluda(nombre):
    return "hola " + nombre


demo = gr.Interface(fn=saluda, inputs="text", outputs="text")

demo.launch()
