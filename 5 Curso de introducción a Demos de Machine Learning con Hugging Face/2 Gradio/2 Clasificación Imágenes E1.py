import tensorflow as tf
import requests
import gradio as gr


def clasifica_imagen(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    print(prediction[:5])
    confidences = {etiquetas[i]: float(prediction[i]) for i in range(1000)}
    return confidences


if __name__ == '__main__':
    inception_net = tf.keras.applications.MobileNetV2()
    ans = requests.get("https://git.io/JJkYN").text
    etiquetas = ans.split("\n")
    print(etiquetas[:10])
    demo = gr.Interface(fn=clasifica_imagen,
                        inputs=gr.Image(shape=(224, 224)),
                        outputs=gr.Label(num_top_classes=3)
                        )

    demo.launch()
