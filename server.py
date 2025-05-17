from flask import Flask, request, jsonify, render_template
from ocr import OCRNeuralNetwork
import numpy as np

# Instancia principal de la aplicación Flask
app = Flask(__name__)
ocr = OCRNeuralNetwork([400, 100, 10])  # Entrada, oculto, salida

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


if __name__ == "__main__":
    app.run(debug=True)
