import io
import os
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Cauli-leaf_resnet18_model.pth"
MAX_FILE_SIZE_MB = 20

# 1. Recreate the exact model structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Black Rot", "Healthy"]  # Must match training order alphabetically

brain = models.resnet18()
num_features = brain.fc.in_features
brain.fc = nn.Linear(num_features, len(class_names))


def load_weights(model, path, target_device):
    """Load PyTorch weights safely across PyTorch versions."""
    try:
        state_dict = torch.load(path, map_location=target_device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=target_device)
    model.load_state_dict(state_dict)


# 2. Load the trained weights once at startup
model_loaded = False
load_error = None

if MODEL_PATH.exists():
    try:
        load_weights(brain, MODEL_PATH, device)
        brain = brain.to(device)
        # 3. Set the model to evaluation mode
        brain.eval()
        model_loaded = True
    except Exception as exc:
        load_error = f"Failed to load model weights: {exc}"
else:
    load_error = f"Model file not found at {MODEL_PATH}"

# Standard inference transform (MUST match validation normalization)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_single_image(image_source, model, transform, labels):
    """Accepts either a file path or a file-like byte stream and returns prediction details."""
    image = Image.open(image_source).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    probs_list = probabilities.detach().cpu().tolist()
    scores = [
        {"label": label, "score": round(float(score), 6)}
        for label, score in zip(labels, probs_list)
    ]
    scores.sort(key=lambda item: item["score"], reverse=True)

    return {
        "predicted_class": labels[predicted_idx.item()],
        "confidence": float(confidence.item()),
        "scores": scores,
        "inference_ms": round(elapsed_ms, 1),
        "device": str(device),
    }


# 4. Flask web server integration
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


@app.after_request
def add_cors_headers(response):
    """Allow local testing even if index.html is opened directly from disk."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": f"Image exceeds the {MAX_FILE_SIZE_MB} MB limit."}), 413


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "CropIntel.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "error": load_error,
        "architecture": "ResNet-18 (PyTorch)",
        "device": str(device),
        "classes": class_names,
    }), (200 if model_loaded else 503)


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    if not model_loaded:
        return jsonify({"error": load_error or "Model is not loaded."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided under the 'image' field."}), 400

    uploaded_file = request.files["image"]
    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"error": "Please select an image file to upload."}), 400

    try:
        image_bytes = io.BytesIO(uploaded_file.read())
        result = predict_single_image(image_bytes, brain, test_transform, class_names)
        return jsonify(result), 200
    except UnidentifiedImageError:
        return jsonify({"error": "Unsupported or corrupted image file. Use JPG, PNG, or WebP."}), 400
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500


if __name__ == "__main__":
    if not model_loaded:
        print(f"[WARNING] {load_error}")
    else:
        print(f"[READY] ResNet-18 loaded on {device} | Classes: {class_names}")

    port = int(os.environ.get("PORT", 5000))
    print(f"Serving Crop Intel at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)