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
        st.write("¡Felicidades! Según nosotros, no tiene un tumor cerebral. Aun así le recomendamos encarecidamente que contacte a un médico para asegurarse de que realmente la información sea verídica. En nuestra base de datos contamos con algunos médicos que le podrán verificar la información. A continuación le proporcionamos el link para que pueda enviar sus datos.")
        st.markdown("<a href='https://main-brainly.vercel.app/profile-patient' target='_blank'>Envíar Datos</a>", unsafe_allow_html=True)

    elif classification == 1:
        st.write("Los tumores pituitarios son crecimientos anormales que se desarrollan en la glándula pituitaria. Algunos tumores pituitarios dan como resultado demasiadas hormonas que regulan funciones importantes de su cuerpo. Algunos tumores pituitarios pueden hacer que la glándula pituitaria produzca niveles más bajos de hormonas.")
        st.write("La mayoría de los tumores hipofisarios son crecimientos no cancerosos (benignos) (adenomas). Los adenomas permanecen en la glándula pituitaria o en los tejidos circundantes y no se diseminan a otras partes del cuerpo.")
        st.write("Hay varias opciones para tratar los tumores pituitarios, incluida la extirpación del tumor, el control de su crecimiento y el control de los niveles hormonales con medicamentos. Su médico puede recomendar la observación, o un enfoque de 'esperar y ver'.")
        st.write("Aun así le recomendamos encarecidamente que contacte a un médico para asegurarse de que realmente la información sea verídica. En nuestra base de datos contamos con algunos médicos que le podrán verificar la información. A continuación le proporcionamos el link para que pueda enviar sus datos.")
        st.markdown("<a href='https://main-brainly.vercel.app/profile-patient' target='_blank'>Envíar Datos</a>", unsafe_allow_html=True)

    elif classification == 2:
        st.write("Un meningioma es un tumor que surge de las meninges, las membranas que rodean el cerebro y la médula espinal. Aunque técnicamente no es un tumor cerebral, se incluye en esta categoría porque puede comprimir o apretar el cerebro, los nervios y los vasos adyacentes. El meningioma es el tipo más común de tumor que se forma en la cabeza.")
        st.write("La mayoría de los meningiomas crecen muy lentamente, a menudo durante muchos años sin causar síntomas. Pero a veces, sus efectos en el tejido cerebral, los nervios o los vasos sanguíneos cercanos pueden causar una discapacidad grave.")
        st.write("Los meningiomas ocurren más comúnmente en mujeres y a menudo se descubren a edades más avanzadas, pero pueden ocurrir a cualquier edad.")
        st.write("Debido a que la mayoría de los meningiomas crecen lentamente, a menudo sin signos ni síntomas significativos, no siempre requieren tratamiento inmediato y pueden controlarse con el tiempo.")
        st.write("Aun así le recomendamos encarecidamente que contacte a un médico para asegurarse de que realmente la información sea verídica. En nuestra base de datos contamos con algunos médicos que le podrán verificar la información. A continuación le proporcionamos el link para que pueda enviar sus datos.")
        st.markdown("<a href='https://main-brainly.vercel.app/profile-patient' target='_blank'>Envíar Datos</a>", unsafe_allow_html=True)

    elif classification == 3:
        st.write("Se detectó un tumor glioma en la imagen.")
        st.write("El glioma es un tipo de tumor que se presenta en el cerebro y la médula espinal. Los gliomas comienzan en las células de apoyo pegajosas (células gliales) que rodean las células nerviosas y las ayudan a funcionar.")
        st.write("Tres tipos de células gliales pueden producir tumores. Los gliomas se clasifican según el tipo de célula glial involucrada en el tumor, así como las características genéticas del tumor, lo que puede ayudar a predecir cómo se comportará el tumor con el tiempo y los tratamientos que tienen más probabilidades de funcionar.")
        st.write("Los tipos de glioma incluyen:")
        st.write("Astrocitomas, incluidos astrocitoma, astrocitoma anaplásico y glioblastoma.")
        st.write("Ependimomas, incluidos el ependimoma anaplásico, el ependimoma mixopapilar y el subependimoma.")
        st.write("Oligodendrogliomas, incluidos oligodendroglioma, oligodendroglioma anaplásico y oligoastrocitoma anaplásico.")
        st.write("Un glioma puede afectar su función cerebral y poner en peligro la vida según su ubicación y la tasa de crecimiento. Los gliomas son uno de los tipos más comunes de tumores cerebrales primarios.")
        st.write("El tipo de glioma que tiene ayuda a determinar su tratamiento y su pronóstico. En general, las opciones de tratamiento del glioma incluyen cirugía, radioterapia, quimioterapia, terapia dirigida y ensayos clínicos experimentales.")
        st.write("Aun así le recomendamos encarecidamente que contacte a un médico para asegurarse de que realmente la información sea verídica. En nuestra base de datos contamos con algunos médicos que le podrán verificar la información. A continuación le proporcionamos el link para que pueda enviar sus datos.")
        st.markdown("<a href='https://main-brainly.vercel.app/profile-patient' target='_blank'>Envíar Datos</a>", unsafe_allow_html=True)

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
