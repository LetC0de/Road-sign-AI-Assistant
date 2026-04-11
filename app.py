from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model 
import os

# LLM imports
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ─── LOAD ML MODEL (ONLY ONCE) ───
model = load_model("best_model.keras")
IMG_SIZE = 32

# ─── LABELS ───
labels = {
    0:"Speed Limit 20", 1:"Speed Limit 30", 2:"Speed Limit 50",
    3:"Speed Limit 60", 4:"Speed Limit 70", 5:"Speed Limit 80",
    6:"End Speed Limit 80", 7:"Speed Limit 100", 8:"Speed Limit 120",
    9:"No Overtaking", 10:"No Overtaking Trucks", 11:"Priority at Intersection",
    12:"Priority Road", 13:"Yield", 14:"Stop", 15:"No Vehicles",
    16:"No Trucks", 17:"No Entry", 18:"General Danger",
    19:"Curve Left", 20:"Curve Right", 21:"Double Curve",
    22:"Bumpy Road", 23:"Slippery Road", 24:"Road Narrows",
    25:"Road Work", 26:"Traffic Signals", 27:"Pedestrian Crossing",
    28:"Children Crossing", 29:"Bicycle Crossing", 30:"Snow",
    31:"Animals Crossing", 32:"End Restrictions", 33:"Turn Right",
    34:"Turn Left", 35:"Go Straight", 36:"Straight or Right",
    37:"Straight or Left", 38:"Keep Right", 39:"Keep Left",
    40:"Roundabout", 41:"End No Overtaking", 42:"End No Overtaking Trucks"
}

class_indices = {"0": 0, "1": 1, "10": 2, "11": 3, "12": 4, "13": 5, "14": 6,
"15": 7, "16": 8, "17": 9, "18": 10, "19": 11, "2": 12, "20": 13, "21": 14,
"22": 15, "23": 16, "24": 17, "25": 18, "26": 19, "27": 20, "28": 21,
"29": 22, "3": 23, "30": 24, "31": 25, "32": 26, "33": 27, "34": 28,
"35": 29, "36": 30, "37": 31, "38": 32, "39": 33, "4": 34, "40": 35,
"41": 36, "42": 37, "5": 38, "6": 39, "7": 40, "8": 41, "9": 42}

label_map = {v: labels[int(k)] for k, v in class_indices.items()}

# ─── LLM SETUP ───
llm = ChatMistralAI(model="mistral-small-2506")

prompt_template = PromptTemplate.from_template("""
You are an expert traffic assistant AI.

A road sign has been detected with the label: "{sign_name}".

Provide a detailed explanation including:
1. Meaning
2. Driver Action
3. Consequences
4. Importance
""")

# ─── HOME ROUTE ───
@app.route("/")
def home():
    return "🚀 Traffic Sign AI Backend Running"



@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if file exists
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # Read image
        img = Image.open(file).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Preprocess
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        result = model.predict(img_array)
        predicted_class = int(np.argmax(result))
        confidence = float(np.max(result)) * 100

        prediction = label_map[predicted_class]

        # Return response
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ─── RUN APP ───
if __name__ == "__main__":
    app.run(debug=True)