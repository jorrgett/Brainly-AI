from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow import keras
from werkzeug.utils import secure_filename
from PIL import Image
import cloudinary
import cloudinary.uploader
import cloudinary.api

app = Flask(__name__)
CORS(app)

model = keras.models.load_model("classification.h5")

classes = ['Ningún Tumor', 'Tumor Pituitario', 'Tumor Meningioma', 'Tumor Glioma']


result = ""
ses = False

def upload_image(file):
    cloudinary.config(
        cloud_name="brainlypf",
        api_key="143982914773545",
        api_secret="Qt7iifjrFNj2-rFkrn9dssdYaME"
    )

    upload_data = cloudinary.uploader.upload(file)
    image_url = upload_data['secure_url']

    return image_url


def names(number):
    if (number == 0):
        return classes[0]
    elif (number == 1):
        return classes[1]
    elif (number == 2):
        return classes[2]
    elif (number == 3):
        return classes[3]


@app.route("/detection", methods=["POST"])
def mainPage():
    if 'file' not in request.files:
        return jsonify(error="Archivo no encontrado")

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="Nombre de archivo no encontrado")

    filename = secure_filename(file.filename)
    img_url = upload_image(file)
    img = Image.open(file)
    dim = (150, 150)
    x = np.array(img.resize(dim))
    x = x.reshape(1, 150, 150, 3)
    answ = model.predict_on_batch(x)
    classification = np.where(answ == np.amax(answ))[1][0]
    predicted_results = names(classification)+' Detectado'

    # Devuelve el resultado de la clasificación y la URL de la imagen en formato JSON
    return jsonify(filename=filename, img_url=img_url, predicted_results=predicted_results)

if __name__ == '__main__':
    app.run()
