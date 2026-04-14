from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model #type:ignore
from flask_cors import CORS
import numpy as np
from PIL import Image
import os

# LLM imports
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

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

# Output parser to format LLM responses
output_parser = StrOutputParser()

# Prompt template for traffic sign predictions with structured output
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are a professional road sign assistant.

STRICT OUTPUT FORMAT:

Description:
[Write 60 - 100 word short sentences about the sign]

Meaning:
- [First key point]
- [Second key point]

Driver Action:
- [What driver should do - point 1]
- [What driver should do - point 2]

Consequences:
- [What happens if ignored - point 1]
- [What happens if ignored - point 2]

Importance:
- [Why it matters - point 1]
- [Why it matters - point 2]

RULES:
- Keep each bullet point under 20 - 40 words
- Use simple, clear language
- No markdown symbols like ** or ##
- Follow the exact format above
"""),

    ("human", "Traffic sign: {sign_name}")
])

# ─── HOME ROUTE ───
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check image
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # ─── IMAGE PROCESSING ───
        img = Image.open(file).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ─── ML PREDICTION ───
        result = model.predict(img_array)
        predicted_class = int(np.argmax(result))
        confidence = float(np.max(result)) * 100

        prediction = label_map[predicted_class]

        # ─── LLM CALL WITH STRUCTURED OUTPUT ───
        # Create a chain: prompt -> llm -> output parser
        chain = prompt_template | llm | output_parser

        # Invoke the chain with the sign name
        explanation = chain.invoke({
            "sign_name": prediction
        })

        # ─── FINAL RESPONSE ───
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ─── CHAT ROUTE ───
# This route handles text messages from the user and sends them to the LLM
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract the user's message from the JSON data
        user_message = data.get("message", "")

        # Validate that message is not empty
        if not user_message or user_message.strip() == "":
            return jsonify({"error": "No message provided"}), 400

        # Create a structured prompt for the LLM with the user's message
        # This ensures the response is well-formatted
        from langchain_core.prompts import ChatPromptTemplate

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a Road sign assistant.

OUTPUT FORMAT:

Answer:
- [First point - keep under 15 words]
- [Second point - keep under 15 words]
- [Third point - keep under 15 words]

RULES:
- Use ONLY bullet points
- No paragraphs or extra text
- No markdown symbols (no **, no ##)
- Keep each point clear and concise
- Maximum 4-5 bullet points
"""),

    ("human", "{question}")
])

        # Create a chain: prompt -> llm -> output parser for structured output
        chat_chain = chat_prompt | llm | output_parser

        # Invoke the chain with the user's message
        response_text = chat_chain.invoke({
            "question": user_message
        })

        # Return the LLM response as JSON
        return jsonify({
            "response": response_text,
            "user_message": user_message
        })

    except Exception as e:
        # Handle any errors and return error message
        return jsonify({"error": str(e)}), 500

# ─── RUN APP ───
if __name__ == "__main__":
    app.run(debug=True)