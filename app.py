import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import pickle
import os
from flask import Flask, request, jsonify

# Load VGG16 model (outputs 4096-dimensional features)
base_model = VGG16(weights="imagenet")
cnn_model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

# Load trained image captioning model
caption_model = load_model("image_caption_model.h5")

# Load tokenizer used for training
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Maximum caption length used during training
MAX_LENGTH = 34


def preprocess_image(img_path):
    """Preprocess image for VGG16 model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def extract_features(img_path):
    """Extracts features from an image using VGG16."""
    img = preprocess_image(img_path)
    features = cnn_model.predict(img)
    return features


def generate_caption(image_features):
    """Generate a caption for an image using the trained model."""
    start_seq = "<start>"
    for _ in range(MAX_LENGTH):
        seq = tokenizer.texts_to_sequences([start_seq])[0]
        seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=MAX_LENGTH)

        # Predict the next word
        pred = caption_model.predict([image_features, seq])
        word_index = np.argmax(pred)
        word = next((word for word, index in tokenizer.word_index.items() if index == word_index), None)

        if word is None or word == "<end>":
            break

        start_seq += " " + word

    return start_seq.replace("<start>", "").strip()


# Initialize Flask API
app = Flask(__name__)


@app.route("/generate_caption", methods=["POST"])
def generate_caption_api():
    """API Endpoint to generate image captions."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided", "success": False})

    image_file = request.files["image"]
    image_path = "temp.jpg"
    image_file.save(image_path)

    try:
        # Extract features
        image_features = extract_features(image_path)

        # Generate caption
        caption = generate_caption(image_features)
        return jsonify({"caption": caption, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False})


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
