from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2
import joblib
import os
import gdown

app = Flask(__name__)

# Ensure the static/uploads directory exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------- Load Models ----------------------------
# Load Crop Recommendation Model
crop_model = joblib.load("gaussiannb_crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

#urls

dense_url = "https://drive.google.com/uc?id=1RJKkFavESsJG5nvM1_Z1CQTtc3hqxL-e"
inception_url = "https://drive.google.com/uc?id=1ifZ4E0ouUmcNylNbVflHIJ4u38nS1TTX"
vgg_url = "https://drive.google.com/uc?id=1gwsV2q8gLfoEJtM78gk1IRbmu_NtmIuQ"

dense_path = "densemodel.h5"
inception_path = "inception_model.h5"
vgg_path = "vgg_model.h5"

if not os.path.exists(dense_path):
    gdown.download(dense_url, dense_path, quiet=False)
    print("dense Model downloaded successfully!")
if not os.path.exists(inception_path):
    gdown.download(inception_url, inception_path, quiet=False)
    print("inception Model downloaded successfully!")
if not os.path.exists(vgg_path):
    gdown.download(vgg_url, vgg_path, quiet=False)
    print("vgg Model downloaded successfully!")

# Load Leaf Disease Detection Models


vgg_model1 = tf.keras.models.load_model(vgg_path)
vgg_model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

inception_model1 = tf.keras.models.load_model(inception_path)
inception_model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

densenet_model1 = tf.keras.models.load_model(dense_path)
densenet_model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Class mapping for leaf diseases
class_mapping = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Blueberry___healthy",
    5: "Cherry_(including_sour)___Powdery_mildew",
    6: "Cherry_(including_sour)___healthy",
    7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    8: "Corn_(maize)___Common_rust_",
    9: "Corn_(maize)___Northern_Leaf_Blight",
    10: "Corn_(maize)___healthy",
    11: "Grape___Black_rot",
    12: "Grape___Esca_(Black_Measles)",
    13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    14: "Grape___healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)",
    16: "Peach___Bacterial_spot",
    17: "Peach___healthy",
    18: "Pepper,_bell___Bacterial_spot",
    19: "Pepper,_bell___healthy",
    20: "Potato___Early_blight",
    21: "Potato___Late_blight",
    22: "Potato___healthy",
    23: "Raspberry___healthy",
    24: "Soybean___healthy",
    25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch",
    27: "Strawberry___healthy",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___Late_blight",
    31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot",
    33: "Tomato___Spider_mites Two-spotted_spider_mite",
    34: "Tomato___Target_Spot",
    35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus",
    37: "Tomato___healthy",
}

# ---------------------------- Helper Functions ----------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize for model compatibility
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def ensemble_prediction(image_path):
    img = preprocess_image(image_path)

    # Get predictions from each model
    vgg_pred = vgg_model1.predict(img)
    inception_pred = inception_model1.predict(img)
    densenet_pred = densenet_model1.predict(img)

    # Average the predictions (ensemble technique)
    final_pred = (vgg_pred + inception_pred + densenet_pred) / 3
    predicted_class = np.argmax(final_pred, axis=1)
    predicted_class_index = predicted_class[0]  # Get the scalar value
    predicted_class_name = class_mapping.get(predicted_class_index, "Unknown Class")
    return predicted_class_name

# ---------------------------- Flask Routes ----------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sample2")
def crop_recommendation():
    return render_template("sample2.html")

@app.route("/leaf_det")
def leaf_detection():
    return render_template("leaf_det.html")

# Crop Recommendation Prediction
@app.route("/predict", methods=["POST"])
def predict_crop():
    try:
        print(request.form)  # Debugging line
        N = float(request.form["nitrogen"])
        P = float(request.form["phosphorus"])
        K = float(request.form["potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = crop_model.predict(input_data)
        crop_name = label_encoder.inverse_transform(prediction)[0]

        return render_template("sample2.html", prediction=crop_name)
    except Exception as e:
        return f"Error: {e}", 400  # Show error message properly

# Leaf Disease Detection Prediction
@app.route("/predict_leaf", methods=["POST"])
def predict_leaf():
    if "leaf_image" not in request.files:
        return "No file uploaded", 400

    file = request.files["leaf_image"]
    if file.filename == "":
        return "No selected file", 400

    # Save the uploaded image
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Get prediction
    prediction = ensemble_prediction(file_path)

    return render_template("leaf_det.html", prediction=prediction)

# ---------------------------- Run the App ----------------------------
#if __name__ == "__main__":
#    app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
