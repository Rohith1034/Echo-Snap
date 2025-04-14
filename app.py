import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import pickle
import os
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras import backend as K

# Constants
MAX_LENGTH = 34

# Load tokenizer and models globally
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

cnn_model = Model(
    inputs=VGG16(weights="imagenet").input,
    outputs=VGG16(weights="imagenet").get_layer("fc2").output
)
caption_model = load_model("image_caption_model.h5")

app = Flask(__name__)

def preprocess_image_from_bytes(img_data):
    """Preprocess image bytes for VGG16 model."""
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def generate_caption(image_features):
    """Generate caption using the trained model."""
    start_seq = "<start>"
    for _ in range(MAX_LENGTH):
        seq = tokenizer.texts_to_sequences([start_seq])[0]
        seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=MAX_LENGTH)

        pred = caption_model.predict([image_features, seq], verbose=0)
        word_index = np.argmax(pred)
        word = next((w for w, idx in tokenizer.word_index.items() if idx == word_index), None)

        if word is None or word == "<end>":
            break

        start_seq += " " + word

    return start_seq.replace("<start>", "").strip()

@app.route("/generate_caption", methods=["POST"])
def generate_caption_api():
    data = request.get_json()

    if not data or "image_data" not in data:
        return jsonify({"error": "Missing 'image_data' in request", "success": False})

    try:
        # Decode base64 image
        img_bytes = base64.b64decode(data["image_data"])
        img_array = preprocess_image_from_bytes(img_bytes)

        # Extract features and generate caption
        image_features = cnn_model.predict(img_array)
        caption = generate_caption(image_features)

        return jsonify({"caption": caption, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

@app.route("/")
def home():
    return "Hello, World!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8070))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
