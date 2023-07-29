from transformers import AutoTokenizer, pipeline
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
# task: que trabajo estará realizando nuestro modelo
# trust_remote_code: es porque en esta ocasión estamos empleando un modelo que no pertenece directamente a los
# `transformers`de HugginFace, entonces es darle permiso de acceder a un modelo ajeno a HF.
# device_map: se usa en conjunto a la biblioteca `accelerate` para buscar la configuración más óptima de Hardware para
# correr nuestros procesos.
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
print("*"*64)
print(type(pipeline))

from langchain import HuggingFacePipeline

llm_falcon = HuggingFacePipeline(
    pipeline=pipeline,
    model_kwargs={
        'temperature': 0,
        'max_length': 200,
        'do_sample': True,  # generá un sampleo aleatorio del texto en diferentes momentos
        'top_k': 10,  # numero de candidatos que va a evaluar el modelo, para decidir cuál es el mejor.
        'num_return_sequences': 1,  # cantidad de respuestas a generar
        'eos_token_id': tokenizer.eos_token_id  # eos = end of sentence, viene dado por el tokenizador que ya hemos usado
    }
)
print("*"*64)
print(llm_falcon)

ans = llm_falcon("What is AI?")
print("*"*64)
print(ans)
