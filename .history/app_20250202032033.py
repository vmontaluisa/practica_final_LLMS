import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


# Cargar el modelo y el tokenizador guardado
model = GPT2LMHeadModel.from_pretrained("gen_modelo_soportev1")
tokenizer = GPT2Tokenizer.from_pretrained("gen_tokenizerv1")


# Configuración del modelo y dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Función para generar texto de prueba
def generate_text(prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Generar texto con el modelo
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Título de la aplicación
st.title("Asistente Virtual de Soporte al Cliente")

# Ingreso de texto por parte del usuario
user_input = st.text_input("¿Cómo puedo ayudarte hoy?", "")

# Mostrar la respuesta generada cuando el usuario ingresa una pregunta
if user_input:
    generated_response = generate_text(user_input)
    st.write(f"**Respuesta generada:** {generated_response}")