from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Cargar el modelo
MODEL_PATH = "modelo_mobilenetv2.h5"
model = load_model(MODEL_PATH)

# Mapeo de clases a nombres (ajusta según tu modelo)
CLASS_NAMES = {
    0: "naranja",
    1: "papa",
    2: "pizza",
    3: "tomate"
}

# Página HTML incrustada (igual que antes)
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Clasificador desde Cámara</title>
</head>
<body style="text-align:center; font-family:Arial;">
    <h1>Clasificación en Tiempo Real</h1>
    <video id="video" width="300" height="225" autoplay></video>
    <br>
    <button onclick="takePhoto()">Tomar Foto</button>
    <h3 id="result"></h3>
    <canvas id="canvas" style="display:none;"></canvas>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const result = document.getElementById('result');

        // Pedir acceso a la cámara
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                alert("Error al acceder a la cámara: " + err);
            });

        function takePhoto() {
            const context = canvas.getContext('2d');
            canvas.width = 224;
            canvas.height = 224;
            context.drawImage(video, 0, 0, 224, 224);

            const dataURL = canvas.toDataURL('image/jpeg');
            fetch('/predict', {
                method: 'POST',
                body: JSON.stringify({ image: dataURL }),
                headers: { 'Content-Type': 'application/json' }
            })
            .then(res => res.json())
            .then(data => {
                result.innerText = "Predicción: " + data.prediction;
            })
            .catch(err => {
                result.innerText = "Error al enviar la imagen: " + err;
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_PAGE

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir imagen en base64
        data = request.get_json()
        image_data = data['image'].split(",")[1]
        img_bytes = base64.b64decode(image_data)

        # Convertir a imagen OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # Asegurar que tenga 3 canales (eliminar alfa si existe)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Redimensionar y preprocesar
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Predicción
        preds = model.predict(img)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx]
        
        # Obtener nombre de la clase
        class_name = CLASS_NAMES.get(class_idx, f"Clase {class_idx}")
        prediction = f"{class_name} (confianza: {confidence:.2f})"

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)