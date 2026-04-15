from flask import Flask, request, jsonify, render_template, session
from tensorflow.keras.models import load_model #type:ignore
from flask_cors import CORS
import numpy as np
from PIL import Image
import os

# LLM imports
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

app.secret_key = os.getenv('SECRET_KEY')

conversation_memory = {}

# ─── LOAD ML MODEL ───
model = load_model("traffic_sign_model.keras")
IMG_SIZE = 64

# ✅ DIRECT CLASS INDICES (NO FILE LOAD)
class_indices = {
 "0": 0, "1": 1, "10": 2, "11": 3, "12": 4, "13": 5, "14": 6,
 "15": 7, "16": 8, "17": 9, "18": 10, "19": 11, "2": 12,
 "20": 13, "21": 14, "22": 15, "23": 16, "24": 17, "25": 18,
 "26": 19, "27": 20, "28": 21, "29": 22, "3": 23, "30": 24,
 "31": 25, "32": 26, "33": 27, "34": 28, "35": 29, "36": 30,
 "37": 31, "38": 32, "39": 33, "4": 34, "40": 35, "41": 36,
 "42": 37, "5": 38, "6": 39, "7": 40, "8": 41, "9": 42
}

# ✅ Reverse mapping (index → folder)
class_labels = {v: k for k, v in class_indices.items()}

# ─── LABEL MAP ───
labels = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicle > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicle > 3.5 tons'
}

# ─── LLM SETUP ───
llm = ChatMistralAI(model="mistral-small-2506")
output_parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a professional road sign assistant."),
    ("human", "Traffic sign: {sign_name}")
])

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'session_id' not in session:
            import uuid
            session['session_id'] = str(uuid.uuid4())

        session_id = session['session_id']

        if session_id not in conversation_memory:
            conversation_memory[session_id] = []

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # Image processing
        img = Image.open(file).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        result = model.predict(img_array)
        predicted_class = int(np.argmax(result))
        confidence = float(np.max(result)) * 100

        # ✅ FIX (same logic, no file loading)
        predicted_folder = class_labels[predicted_class]
        prediction = labels[int(predicted_folder)]

        # LLM
        chain = prompt_template | llm | output_parser

        explanation = chain.invoke({
            "sign_name": prediction
        })

        conversation_memory[session_id].append(
            HumanMessage(content=f"User clicked on traffic sign: {prediction}")
        )
        conversation_memory[session_id].append(
            AIMessage(content=explanation)
        )

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)