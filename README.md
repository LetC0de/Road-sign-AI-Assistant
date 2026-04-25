# 🚦 Road Sign AI Assistant – ML + LLM Powered Intelligent System

An end-to-end AI application that detects traffic signs from images and provides intelligent, structured explanations using Deep Learning and Large Language Models. The system also supports conversational interaction with memory for a smarter user experience.

---

## 🌐 Live Demo

* 🚀 Frontend (Vercel): https://road-sign-ai-assistant.vercel.app
* ⚙️ Backend (Render): https://road-sign-ai-assistant.onrender.com

---

## 🚀 Features

* 📷 Upload traffic sign images for detection
* 🧠 CNN-based classification (43 classes)
* 🎯 Real-time prediction with confidence score
* 🤖 AI-generated structured explanation using LLM
* 💬 Chat system with conversation memory
* 🖼️ Interactive traffic sign gallery UI
* 🔁 Session-based user interaction tracking
* ⚡ Fast API with Flask backend

---

## 🧠 Tech Stack

### 🔹 Frontend

* HTML, CSS, JavaScript
* Interactive UI with image upload + chat
* Dynamic traffic sign gallery
* Deployed on Vercel

### 🔹 Backend

* Python
* Flask
* Flask-CORS
* Session management

### 🔹 Machine Learning

* TensorFlow / Keras
* CNN (Traffic Sign Classification Model)
* NumPy
* PIL (Image Processing)

### 🔹 AI / LLM

* LangChain
* Mistral AI (ChatMistralAI)
* Prompt Engineering
* Conversation Memory

### 🔹 Utilities

* python-dotenv
* Garbage Collection (Memory Optimization)

---

## 📊 Dataset

* **German Traffic Sign Recognition Benchmark (GTSRB)**
* 📁 43 traffic sign classes
* 🖼️ Thousands of labeled images
* Used to train the CNN model from scratch

---

## 🏗️ Model Training

The project includes a `model_training.py` file to train the model from scratch:

* Image preprocessing (resize, normalization)
* CNN architecture using Keras
* Training on GTSRB dataset
* Model saved as `.keras` file

---

## 🎨 Frontend UI

* 📂 Image upload system
* 🖼️ Traffic sign gallery (click-to-analyze)
* 💬 Chat interface with LLM
* 🔔 Real-time notifications
* ⚡ Dynamic response rendering

---

## 🔄 System Workflow

```text
User uploads image / selects sign
        ↓
Image preprocessing (resize, normalize)
        ↓
CNN Model → Predict traffic sign
        ↓
Prediction + confidence returned
        ↓
User clicks "Explain"
        ↓
LangChain Prompt → Mistral LLM
        ↓
Structured explanation generated
        ↓
Optional chat with memory
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/LetC0de/Road-sign-AI-Assistant.git
cd Project
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key
SECRET_KEY=your_secret_key
```

---

## ▶️ Usage

Run backend:

```bash
python app.py
```

Open frontend locally or deploy on Vercel.

---

## 📁 Project Structure

```text
road-sign-ai-assistant/
│
├── app.py
├── model_training.py
├── traffic_sign_model.keras
├── images/
├── static/
├── templates/
├── frontend/ (HTML, CSS, JS)
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚠️ Notes

* Model file must be present before running
* `.env` is ignored for security
* Sessions auto-expire after 30 minutes
* Optimized for low-memory deployment (Render free tier)


## 👨‍💻 Author

Abhishek Nishad

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
