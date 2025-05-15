const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.clientWidth, canvas.height);

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
    const imgData = ctx.getImageData(0, 0, 200, 200).data;
    const resized = new Array(400).fill(0); // 20x20
    // Código para reducir a 20x20 y pasar a escala de grises
    return resized;
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