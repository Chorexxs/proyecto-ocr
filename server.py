from flask import Flask, request, jsonify, render_template
from ocr import OCRNeuralNetwork
import numpy as np
import os

# Instancia principal de la aplicación Flask
app = Flask(__name__)
ocr = OCRNeuralNetwork([400, 100, 10])  # Entrada, oculto, salida

# Ruta absoluta para pesos.pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, "pesos.pkl")

# Carga los pesos si existen
if os.path.exists(weights_path):
    ocr.load(weights_path)
    print("Pesos cargados correctamente.")

# Ruta principal que renderiza la interfaz HTML


@app.route("/")
def index():
    return render_template("index.html")

# Ruta que maneja las solicitudes de predicción (POST)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Extrae el JSON enviado por el cliente
    input_vector = np.array(data["input"]).reshape(
        (400, 1))  # Convierte el input en un vector columna
    result = ocr.predict(input_vector)  # Ejecuta la predicción
    # Retorna la predicción como JSON
    return jsonify({"prediction": int(result)})

# Ruta que maneja las solicitudes de entrenamiento (POST)


@app.route("/train", methods=["POST"])
def train_model():
    data = request.get_json()
    input_vector = np.array(data["input"]).reshape((400, 1))
    label = int(data["label"])

    y = np.zeros((10, 1))
    y[label] = 1.0

    # Entrena el modelo con el nuevo dato
    ocr.train([(input_vector, y)], epochs=5, learning_rate=0.5)
    ocr.save(weights_path)
    return jsonify({"status": "entrenado"})


@app.route("/save", methods=["POST"])
def save_model():
    ocr.save(weights_path)
    return jsonify({"status": "guardado"})


if __name__ == "__main__":
    app.run(debug=True)
