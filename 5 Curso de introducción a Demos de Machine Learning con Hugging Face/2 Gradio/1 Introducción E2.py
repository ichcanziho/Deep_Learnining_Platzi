import gradio as gr


def saluda(nombre):
    return "hola " + nombre


demo = gr.Interface(fn=saluda, inputs=gr.Textbox(lines=10, placeholder="Dime tu nombre porfa"), outputs="text")

demo.launch()
