import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import pickle
import os
from flask import Flask, request, jsonify
from tensorflow.keras import backend as K

# Constants
MAX_LENGTH = 34

# Load tokenizer once
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Initialize Flask
app = Flask(__name__)


def preprocess_image(img_path):
    """Preprocess image for VGG16 model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def get_cnn_model():
    """Load VGG16 model that outputs 4096-dimensional features."""
    base_model = VGG16(weights="imagenet")
    return Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)


@tf.function
def extract_features(img_array, cnn_model):
    """Extracts features using VGG16."""
    return cnn_model(img_array, training=False)


def generate_caption(image_features, caption_model):
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
    if "image" not in request.files:
        return jsonify({"error": "No image file provided", "success": False})

    image_file = request.files["image"]
    image_path = "temp.jpg"
    image_file.save(image_path)

    try:
        # Load models on demand
        cnn_model = get_cnn_model()
        caption_model = load_model("image_caption_model.h5")

        # Preprocess and extract features
        img_array = preprocess_image(image_path)
        image_features = cnn_model.predict(img_array)

        # Generate caption
        caption = generate_caption(image_features, caption_model)

        # Clear memory
        K.clear_session()
        del cnn_model, caption_model

        return jsonify({"caption": caption, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

@app.route("/")
def home():
    return "Hello, World!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9090))
    app.run(host="0.0.0.0", port=port)

