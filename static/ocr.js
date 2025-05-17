const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let painting = false;

canvas.addEventListener("mousedown", () => painting = true);
canvas.addEventListener("mouseup", () => painting = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!painting) return;
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(e.offsetX, e.offsetY, 10, 0, 2 * Math.PI);
    ctx.fill();
}

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function getImageData() {
    // Obtiene los datos del canvas
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = 20;
    srcCanvas.height = 20;
    const srcCtx = srcCanvas.getContext("2d");

    // Escala la imagen original a 20x20
    srcCtx.drawImage(canvas, 0, 0, 20, 20);
    const imgData = srcCtx.getImageData(0, 0, 20, 20);
    const data = imgData.data;

    // Convertir RGBA a blanco (1) o negro (0)
    const result = [];
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const avg = (r + g + b) / 3;

        // Normalizar; blanco = 1, negro = 0
        const normalized = avg > 128 ? 1 : 0;
        result.push(normalized);
    }

    return result;
}

function predict() {
    const data = getImageData();
    fetch("/predict", {
        method: "POST",
        headers: {"Content-type": "application/json"},
        body: JSON.stringify({input: data})
    })
    .then(res => res.json())
    .then(result => {
        document.getElementById("result").textContent ="Predicción: " + result.prediction;
    });
}

// Guardar y entrenar datos

function saveSample(label) {
    const data = getImageData();
    fetch("/save-sample", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({input: data, label: label})
    }).then(res => res.json())
      .then(res => alert("Guardado. Total de muestras: " + res.samples));
}

function trainModel() {
    fetch("/train", {method: "POST"})
        .then(res => res.json())
        .then(res => alert(res.status));
}

function saveModel() {
    fetch("/save-model", {method: "POST"})
        .then(res => res.json())
        .then(res => alert(res.status));
}

function loadModel() {
    fetch("/load-model", {method: "POST"})
        .then(res => res.json())
        .then(res => alert(res.status));
}