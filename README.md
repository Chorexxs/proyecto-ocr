# 🧠 Proyecto OCR con Flask

!(static/portada.png)

Este es un proyecto de reconocimiento óptico de caracteres (OCR) que permite al usuario dibujar un número del 0 al 9 en un lienzo, y una red neuronal entrenada en Python (usando solo NumPy) se encarga de predecir qué número es. La aplicación utiliza Flask como backend y HTML/JavaScript para el frontend.

---

## Características

- Interfaz interactiva con canvas para dibujar números.
- Predicción de números mediante una red neuronal simple.
- Entrenamiento incremental: puedes mejorar el modelo en tiempo real añadiendo más ejemplos.
- Guardado automático de pesos en `pesos.pkl` tras el entrenamiento.
- Interfaz web simple y funcional.

---

## Requisitos

- Python 3.x
- Flask
- NumPy
- Pickle

Instálalos ejecutando:

```bash
pip install -r requirements.txt
```

---

## Cómo ejecutar el proyecto

1. Clona el repositorio o copia los archivos a una carpeta local.
2. Asegúrate de que `server.py`, `ocr.py` y las carpetas `templates/` y `static/` estén en el mismo directorio.
3. Ejecuta el servidor Flask:

```bash
python server.py
```

4. Abre tu navegador y ve a: [http://localhost:5000](http://localhost:5000)

---

## ¿Cómo funciona?

- **Dibujo:** el usuario dibuja un número en el `canvas`.
- **Procesamiento:** la imagen se reduce a una matriz de 20x20 píxeles, normalizada en valores 0 o 1.
- **Predicción:** se envía al backend, donde la red neuronal devuelve el número más probable.
- **Entrenamiento:** puedes introducir el número correcto y mejorar la red neuronal con tus propios datos.

---

## Guardado y carga de pesos

- Después de entrenar con nuevos datos, los pesos se guardan automáticamente en `pesos.pkl`.
- Al iniciar la app, si el archivo `pesos.pkl` existe, se cargan automáticamente los pesos anteriores.

---

## Nota

Este proyecto es ideal como introducción al aprendizaje automático desde cero sin usar frameworks como TensorFlow o PyTorch. Perfecto para aprender cómo funcionan las redes neuronales por dentro

---

## Autor

Desarrollado con ❤️ por [Chorexxs](https://chorexxs-portfolio.dev/)

Puedes usar, modificar y compartir este proyecto con fines educativos.
