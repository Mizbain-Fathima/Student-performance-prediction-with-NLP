# Student Performance Prediction with AI Explanations 

A Streamlit-based web application that predicts whether a student will **PASS** or **FAIL** based on academic, behavioral, lifestyle, and family factors — and provides **natural-language explanations** and **personalized improvement advice** using an AI agent (Groq LLM).

---

## Features

### **Pass/Fail Prediction**
- Predicts student outcomes using a trained **Decision Tree Model**.
- Uses features from the **Student Performance Dataset (UCI)**.

### **AI-Generated Explanation (NLP Layer)**

Powered by **llama-3.3**:
- Explains *why* the student is predicted to pass or fail.
- Identifies *key influencing factors*.
- Provides *personalized recommendations* for improvement.
- Produces warm, human-like guidance (counselor style).

### **Interactive Streamlit UI**

- Clean and simple user interface.
- Easy sliders and dropdowns for input.
- Instant prediction + AI explanation.

### **ML Pipeline**

- Preprocessing with One-Hot Encoding & scaling
- DecisionTreeClassifier for prediction
- Feature importance extraction for interpretability

## Project Structure
```bash
student-performance-prediction-with-NLP/
│── app_pass_fail.py # Main Streamlit App
│── best_pass_fail_model.joblib # Trained ML model
│── requirements.txt # Dependencies
│── README.md # Documentation
│── .env.example # Environment variable example
│── models/ # Saved NLP model (optional)
└── data/ # Dataset (optional)
```

## Installation & Local Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Mizbain-Fathima/student-performance-prediction-with-NLP.git
cd student-performance-prediction-with-NLP
```

### 2️⃣ Create & activate a virtual environment
```bash
python -m venv venv
venv/Scripts/activate  # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set up environment variables
Create a .env file:
```bash
GROQ_API_KEY=your_groq_key_here
```

### ▶️ Running the App Locally
```bash
streamlit run app_pass_fail.py
```

App will open at:
http://localhost:8501

## Deployment (Render)

#### 1️⃣ Push to GitHub
Make sure your repo contains:
```bash
requirements.txt
app_pass_fail.py
best_pass_fail_model.joblib
```

### 2️⃣ Go to Render

https://dashboard.render.com
 → New Web Service

### 3️⃣ Enter these settings:

Build Command
```bash
pip install -r requirements.txt

# Start Command
streamlit run app_pass_fail.py --server.port 10000 --server.address 0.0.0.0
```

### 4️⃣ Add Environment Variable
GROQ_API_KEY = your_key_here

