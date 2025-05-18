// Selecciona el canvas del DOM y obtiene su contexto 2D
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
// Rellena el canvas completamente de negro al inicio
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Variable para saber si el usuario está dibujando
let painting = false;

// Eventos de ratón para empezar y detener el dibujo
canvas.addEventListener("mousedown", () => painting = true); // Empieza a dibujar
canvas.addEventListener("mouseup", () => painting = false); // Detiene el dibujo
canvas.addEventListener("mousemove", draw); // Detecta movimiento del mouse

// Función para dibujar círculos blancos donde se mueve el ratón
function draw(e) {
    if (!painting) return;
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(e.offsetX, e.offsetY, 10, 0, 2 * Math.PI);
    ctx.fill();
}

// Limpia completamente el canvas y lo deja negro
function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Extrae y procesa los datos del canvas para enviarlos al servidor
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

// Envía los datos al servidor y muestra el resultado en pantalla
function predict() {
    const data = getImageData();
    fetch("/predict", {
        method: "POST",
        headers: {"Content-type": "application/json"},
        body: JSON.stringify({input: data}) // Envia la imagen procesada
    })
    .then(res => res.json())
    .then(result => {
        // Muestra la predicción en el elemento con id="result"
        document.getElementById("result").textContent ="Predicción: " + result.prediction;
    });
}

// Función de entrenamiento del modelo
function train() {
    const data = getImageData();
    const label = parseInt(document.getElementById("label").value);
    if (isNaN(label) || label < 0 || label > 9) {
        alert("Ingresa un número del 1 al 9");
        return;
    }

    fetch("/train", {
        method: "POST",
        headers: {"Content-type": "application/json"},
        body: JSON.stringify({input: data, label: label})
    })
    .then(res => res.json())
    .then(result => {
        alert("Modelo entrenado con el número " + label);
    });
}

