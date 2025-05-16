from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)
ocr = OCRNeuralNetwork([400, 100, 10])  # Entrada, oculto, salida


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_vector = np.array(data["input"]).reshape((400, 1))
    result = ocr.predict(input_vector)
    return jsonify({"prediction": int(result)})


if __name__ == "__main__":
    app.run(debug=True)
