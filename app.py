from flask import Flask, request, jsonify, render_template, session, send_from_directory
from tensorflow.keras.models import load_model #type:ignore
from flask_cors import CORS
import numpy as np
from PIL import Image
import os

print("🚀 Starting Flask App...")

# LLM imports
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["https://road-sign-ai-assistant.vercel.app"])

# Set secret key for session management
# This is required for Flask sessions to work
app.secret_key = os.getenv('SECRET_KEY')

# Dictionary to store conversation history per session
# Key: session_id, Value: list of messages (HumanMessage and AIMessage)
conversation_memory = {}

# ─── LOAD ML MODEL ───
model_path = os.path.join(os.getcwd(), "traffic_sign_model.h5",compile=False)
if not os.path.exists(model_path):
    raise Exception(f"Model file not found at {model_path}")
model = load_model(model_path)
print("✅ Model loaded successfully")
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

# Output parser to format LLM responses
output_parser = StrOutputParser()

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


# ─── SERVE IMAGES FROM IMAGES FOLDER ───
# This route serves images from the images/ folder for model prediction
# Frontend displays images from static/Meta/ but sends images from images/ folder to model
@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory("images", filename)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get or create session ID for this user
        if 'session_id' not in session:
            import uuid
            session['session_id'] = str(uuid.uuid4())

        session_id = session['session_id']

        # Initialize conversation memory for this session if it doesn't exist
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []

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

        # ✅ FIX: Convert model prediction to correct label
        # Model output → folder name → actual label name
        predicted_folder = class_labels[predicted_class]
        prediction = labels[int(predicted_folder)]

        # ─── LLM CALL WITH STRUCTURED OUTPUT AND MEMORY ───
        # Create a chain: prompt -> llm -> output parser
        chain = prompt_template | llm | output_parser

        # Invoke the chain with the sign name
        explanation = chain.invoke({
            "sign_name": prediction
        })

        # Store this interaction in conversation memory
        # Save what sign was detected and the explanation
        conversation_memory[session_id].append(
            HumanMessage(content=f"User clicked on traffic sign: {prediction}")
        )
        conversation_memory[session_id].append(
            AIMessage(content=explanation)
        )

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
# It maintains conversation memory so the LLM can reference previous messages
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get or create session ID for this user
        if 'session_id' not in session:
            import uuid
            session['session_id'] = str(uuid.uuid4())

        session_id = session['session_id']

        # Initialize conversation memory for this session if it doesn't exist
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []

        # Get JSON data from the request
        data = request.get_json()

        # Extract the user's message from the JSON data
        user_message = data.get("message", "")

        # Validate that message is not empty
        if not user_message or user_message.strip() == "":
            return jsonify({"error": "No message provided"}), 400

        # Create a structured prompt for the LLM with conversation history
        # This allows the LLM to reference previous signs and conversations
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a Road sign assistant with memory of the conversation.

CONTEXT: You can reference previous traffic signs that were discussed in this conversation.
If the user asks about "this sign" or "that sign", refer to the most recent sign mentioned.

OUTPUT FORMAT:

Answer:
   You can give me short description about 100 - 200 words if needed.
- [First point - keep under 50 - 100 words as need based on question]
- [Second point - keep under 50 - 100 words as need based on question]
- [Third point - keep under 50 - 100 words as need based on question]

RULES:
- Use ONLY bullet points
- No paragraphs or extra text
- No markdown symbols (no **, no ##)
- Keep each point clear and concise
- Maximum 4-5 bullet points
- Reference previous context when relevant
"""),
            # Include conversation history here
            *conversation_memory[session_id],
            # Add the current user message
            ("human", "{question}")
        ])

        # Create a chain: prompt -> llm -> output parser for structured output
        chat_chain = chat_prompt | llm | output_parser

        # Invoke the chain with the user's message
        response_text = chat_chain.invoke({
            "question": user_message
        })

        # Store this interaction in conversation memory
        conversation_memory[session_id].append(
            HumanMessage(content=user_message)
        )
        conversation_memory[session_id].append(
            AIMessage(content=response_text)
        )

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)