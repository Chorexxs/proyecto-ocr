from flask import Flask, request, jsonify, render_template
from ocr import OCRNeuralNetwork
import numpy as np

app = Flask(__name__)
ocr = OCRNeuralNetwork([400, 100, 10])  # Entrada, oculto, salida
samples = []  # Aquí se guardan temporalmente los ejemplos


@app.route("/save-sample", methods=["POST"])
def save_sample():
    data = request.get_json()
    input_vector = np.array(data["input"]).reshape((400, 1)).astype(float)
    label = int(data["label"])
    label_vector = np.zeros((10, 1))
    label_vector[label] = 1.0
    samples.append((input_vector, label_vector))
    return jsonify({"status": "ok", "sample": len(samples)})


@app.route("/train", methods=["POST"])
def train_model():
    if not samples:
        return jsonify({"status": "no samples to train"})

    ocr.train(samples, epochs=1000, learning_rate=0.5)
    return jsonify({"status": "training complete", "samples": len(samples)})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_vector = np.array(data["input"]).reshape((400, 1))
    result = ocr.predict(input_vector)
    return jsonify({"prediction": int(result)})


@app.route("/save-model", methods=["POST"])
def save_model():
    ocr.save("model.pkl")
    return jsonify({"status": "model saved"})


@app.route("/load-model", methods="[POST]")
def load_model():
    try:
        ocr.load("model.pkl")
        return jsonify({"status": "model loaded"})
    except:
        return jsonify({"status": "no model found"})


if __name__ == "__main__":
    app.run(debug=True)
