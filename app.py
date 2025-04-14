import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from flask import Flask, request, jsonify
import pickle

# Constants
MAX_LENGTH = 34

# Load tokenizer once
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load VGG16 model (for feature extraction)
def get_cnn_model():
    base_model = VGG16(weights="imagenet")
    return Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

cnn_model = get_cnn_model()

# Load caption generation model
caption_model = load_model("image_caption_model.h5")

# Initialize Flask app
app = Flask(__name__)

def preprocess_image(img_path):
    """Preprocess image for VGG16 model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

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

@app.route("/", methods=["GET"])
def home():
    return "Hello, world!", 200

@app.route("/generate_caption", methods=["GET", "POST"])
def generate_caption_api():
    if request.method == "GET":
        return "Send a POST request with an image to get a caption."

    if "image" not in request.files:
        return jsonify({"error": "No image file provided", "success": False})
    
    image_file = request.files["image"]
    image_path = "temp.jpg"
    image_file.save(image_path)

    try:
        img_array = preprocess_image(image_path)
        image_features = cnn_model.predict(img_array)
        caption = generate_caption(image_features)
        return jsonify({"caption": caption, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False})
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
