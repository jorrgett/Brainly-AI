import streamlit as st
import webbrowser
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from werkzeug.utils import secure_filename
import cloudinary
import cloudinary.uploader
import cloudinary.api
import json

st.set_page_config(page_title="Tumor Detection", page_icon=":microscope:", layout="wide")

model = keras.models.load_model("classification.h5")
classes = ['Ningún Tumor', 'Tumor Pituitario', 'Tumor Meningioma', 'Tumor Glioma']


def names(number):
    if (number == 0):
        return classes[0]
    elif (number == 1):
        return classes[1]
    elif (number == 2):
        return classes[2]
    elif (number == 3):
        return classes[3]


def upload_image(file):
    cloudinary.config(
        cloud_name="brainlypf",
        api_key="143982914773545",
        api_secret="Qt7iifjrFNj2-rFkrn9dssdYaME"
    )

    upload_data = cloudinary.uploader.upload(file)
    image_url = upload_data['secure_url']

    return image_url


def predict(image):
    dim = (150, 150)
    x = np.array(image.resize(dim))
    x = x.reshape(1, 150, 150, 3)
    answ = model.predict_on_batch(x)
    classification = np.where(answ == np.amax(answ))[1][0]

    if classification == 0:
        return "No se detectó ningún tumor en la imagen. ¡Felicidades! Según nosotros, no tiene un tumor cerebral, pero le recomendamos encarecidamente que visite a un médico para asegurarse de que está realmente a salvo. En nuestra base de datos contamos con algunos médicos que le podrán verificar la información."
    elif classification == 1:
        if st.button('Más información sobre Tumor Pituitario'):
            webbrowser.open('https://www.cancer.gov/espanol/tipos/tumor-de-hipofisis')
        return "Se detectó un tumor pituitario en la imagen. Los tumores pituitarios son crecimientos anormales que se desarrollan en la glándula pituitaria. Algunos tumores pituitarios dan como resultado demasiadas hormonas que regulan funciones importantes de su cuerpo. Algunos tumores pituitarios pueden hacer que la glándula pituitaria produzca niveles más bajos de hormonas. Aun así le recomendamos encarecidamente que contacte a un médico para asegurarse de que realmente la información sea verídica. En nuestra base de datos contamos con algunos médicos que le podrán verificar la información."
    elif classification == 2:
        return "Se detectó un tumor meningioma en la imagen. Un meningioma es un tumor que surge de las meninges, las membranas que rodean el cerebro y la médula espinal. Aunque técnicamente no es un tumor cerebral, se incluye en esta categoría porque puede comprimir o apretar el cerebro, los nervios y los vasos adyacentes. Aun así le recomendamos encarecidamente que contacte a un médico para asegurarse de que realmente la información sea verídica. En nuestra base de datos contamos con algunos médicos que le podrán verificar la información."
    elif classification == 3:
        return "Se detectó un tumor glioma en la imagen. El glioma es un tipo de tumor que se presenta en el cerebro y la médula espinal. Los gliomas comienzan en las células de apoyo pegajosas (células gliales) que rodean las células nerviosas y las ayudan a funcionar. Aun así le recomendamos encarecidamente que contacte a un médico para asegurarse de que realmente la información sea verídica. En nuestra base de datos contamos con algunos médicos que le podrán verificar la información."

def main():
    st.title("Detección de tumores cerebrales")

    uploaded_file = st.file_uploader("Cargue una imagen de tomografía computarizada o resonancia magnética", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen cargada', use_column_width=True)
        st.write("")
        st.write("Predicción:")
        result = predict(image)
        st.write(result)

if __name__ == '__main__':
    main()
