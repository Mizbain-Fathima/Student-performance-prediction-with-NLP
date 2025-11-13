import streamlit as st
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
from groq import Groq
import os
load_dotenv()

# Load trained pipeline model
model = joblib.load("best_pass_fail_model.joblib")

st.set_page_config(
    page_title="🎓 Student Pass/Fail Predictor",
    page_icon="📘",
    layout="centered"
)

st.title("🎓 Student Pass/Fail Predictor")
st.write("Predict whether a student is likely to **pass or fail**, with explanations and improvement advice.")

st.markdown("---")
st.subheader("🧍 Student Information")

# =========================
# 1. Extract Top Features
# =========================
def get_top_features(model, input_df):
    """
    Extract top contributing features from a Pipeline with preprocessing + classifier.
    """

    # Get classifier from pipeline
    try:
        clf = model.named_steps["model"]   # DecisionTreeClassifier
    except:
        return []

    if not hasattr(clf, "feature_importances_"):
        return []

    importances = clf.feature_importances_

    # Get encoded feature names from preprocessing step
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
    except:
        feature_names = input_df.columns

    # Pair names and importances
    pairs = list(zip(feature_names, importances))
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    # Map encoded names (e.g. sex_F, reason_home) → base feature name (sex, reason)
    cleaned = [feat.split('_')[0] for feat, score in sorted_pairs[:3]]

    return cleaned


# =========================
# 2. Mappings for Explanations
# =========================
reason_map = {
    'G1': "Your first period grade (G1) was low, indicating weak initial understanding.",
    'G2': "Your second period grade (G2) was low, which strongly affects your final performance.",
    'absences': "You have a high number of absences, reducing classroom continuity.",
    'failures': "You have previous subject failures, indicating foundational gaps.",
    'studytime': "Your weekly study time is low, impacting preparation.",
    'freetime': "Excess free time compared to study time reduces focus.",
    'goout': "Frequent outings reduce study hours and concentration."
}

advice_map = {
    'G1': "Revise earlier topics and solve past papers to strengthen basics.",
    'G2': "Focus on continuous assessments with weekly revision.",
    'absences': "Reduce absences — aim for 90%+ attendance.",
    'failures': "Review weak subjects thoroughly with targeted practice.",
    'studytime': "Increase study time to at least 2 hours/day using Pomodoro technique.",
    'freetime': "Balance free time with structured study routines.",
    'goout': "Limit weekday outings to stay consistent."
}

feature_meaning = {
    "Fedu": "Father’s education level may influence academic support at home.",
    "sex": "Gender pattern in dataset influences pass-rate correlations.",
    "school": "School environment and learning support impact the performance.",
    "G1": "Very low first period grade indicates poor fundamental understanding.",
    "G2": "Poor second period grade strongly predicts final performance.",
    "absences": "High absences reduce consistency in learning.",
    "studytime": "Low study time weakens preparation.",
    "failures": "Previous academic failures indicate learning gaps.",
    "freetime": "More free time reduces academic focus.",
    "goout": "Frequent outings reduce study hours." 
}

# =========================
# 3. Generate Explanations
# =========================
def generate_explanation(top_features, prediction_label):
    reasons = [reason_map.get(f, "") for f in top_features if reason_map.get(f)]

    if prediction_label == 1:
        return "You are likely to PASS.\n\nKey positive factors:\n- " + "\n- ".join(reasons)
    else:
        return "You are likely to FAIL.\n\nKey contributing reasons:\n- " + "\n- ".join(reasons)

MODEL_PATH="llama-3.1-8b-instant"

def generate_advice(top_features):
    tips = [advice_map.get(f, "") for f in top_features if advice_map.get(f)]

    if not tips:
        return "Maintain consistent study habits and focus on continuous improvement."

    return "To improve further:\n- " + "\n- ".join(tips)

def convert_features_to_meaning(feats):
    return [feature_meaning.get(f, f) for f in feats]

def humanize_features(features):
    return [feature_meaning.get(f, f"Impact from {f}") for f in features]

@st.cache_resource
def load_summarizer():
    model_name = "google/flan-t5-base"

    # If local model exists, load locally
    if os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    else:
        # Download once, then save locally
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)

    return tokenizer, model

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_summary(pred, top_features, explanation, advice, prob, new_data):
    result = "PASS" if pred == 1 else "FAIL"

    g1 = new_data["G1"].iloc[0]
    g2 = new_data["G2"].iloc[0]
    absn = new_data["absences"].iloc[0]
    stime = new_data["studytime"].iloc[0]
    fails = new_data["failures"].iloc[0]

    readable = ", ".join(top_features)

    prompt = f"""
You are an expert school counselor. Write a warm, empathetic, 7–10 line explanation about a student's academic outcome based on the following information.

STUDENT OUTCOME: {result}
CONFIDENCE: {prob*100:.1f}%

KEY FACTORS: {readable}

ACADEMIC INFO:
- First Period Grade (G1): {g1}
- Second Period Grade (G2): {g2}
- Total Absences: {absn}
- Weekly Study Time (1–4): {stime}
- Past Failures: {fails}

Write a natural, human-like paragraph that:
- Clearly explains why the student is predicted to {result}
- Describes how grades, study habits, attendance, and family background influence performance
- Acknowledges strengths the student has
- Gives actionable improvement advice
- Avoids repetition or generic statements
- Does NOT mention AI, predictions, models, or probability
- Sounds like a supportive teacher guiding the student
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300
    )

    return response.choices[0].message.content

# =========================
# 4. FORM UI
# =========================
with st.form("student_form"):

     # ===================== Basic Info =====================
    st.markdown("### 📘 General Information")
    school = st.selectbox("School (GP: Gabriel Pereira, MS: Mousinho da Silveira)", ["GP", "MS"])
    sex = st.selectbox("Gender", ["F (Female)", "M (Male"])
    age = st.slider("Age (in years)", 15, 22, 17)
    address = st.selectbox("Address Type (U: Urban, R: Rural)", ["U", "R"])
    famsize = st.selectbox("Family Size (LE3: ≤3 members, GT3: >3 members)", ["LE3", "GT3"])
    Pstatus = st.selectbox("Parent Cohabitation Status (T: Together, A: Apart)", ["T", "A"])

    # ===================== Education =====================
    st.markdown("### 🧠 Education Background")
    Medu = st.slider("Mother’s Education (0: None – 4: Higher Education)", 0, 4, 2)
    Fedu = st.slider("Father’s Education (0: None – 4: Higher Education)", 0, 4, 2)
    Mjob = st.selectbox("Mother’s Job", ["teacher", "health", "services", "at_home", "other"])
    Fjob = st.selectbox("Father’s Job", ["teacher", "health", "services", "at_home", "other"])
    reason = st.selectbox("Reason for Choosing School", ["home", "reputation", "course", "other"])
    guardian = st.selectbox("Main Guardian", ["mother", "father", "other"])

    # ===================== Study Info =====================
    st.markdown("### 📚 Study & Behavior Factors")
    traveltime = st.slider("Travel Time (1: <15min, 2: 15–30min, 3: 30–60min, 4: >60min)", 1, 4, 1)
    studytime = st.slider("Weekly Study Time (1: <2h, 2: 2–5h, 3: 5–10h, 4: >10h)", 1, 4, 2)
    failures = st.slider("Past Class Failures (0–4)", 0, 4, 0)
    absences = st.number_input("Total Absences", 0, 100, 5)

    # ===================== Support & Lifestyle =====================
    st.markdown("### 💬 Support & Lifestyle")
    schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])
    famsup = st.selectbox("Family Support", ["yes", "no"])
    paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
    activities = st.selectbox("Extracurricular Activities", ["yes", "no"])
    nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])
    internet = st.selectbox("Has Internet Access at Home", ["yes", "no"])
    romantic = st.selectbox("In a Relationship", ["yes", "no"])

    # ===================== Grades =====================
    st.markdown("### 🧾 Academic Performance")
    G1 = st.slider("First Period Grade (G1: 0–20)", 0, 20, 12)
    G2 = st.slider("Second Period Grade (G2: 0–20)", 0, 20, 13)
    famrel = st.slider("Family Relationship Quality (1–5)", 1, 5, 4)
    freetime = st.slider("Free Time After School (1–5)", 1, 5, 3)
    goout = st.slider("Going Out Frequency (1–5)", 1, 5, 2)
    Dalc = st.slider("Workday Alcohol Consumption (1–5)", 1, 5, 1)
    Walc = st.slider("Weekend Alcohol Consumption (1–5)", 1, 5, 1)
    health = st.slider("Current Health Status (1–5)", 1, 5, 4)

    submitted = st.form_submit_button("🎯 Predict Pass/Fail")
    
# =========================
# 5. Prediction + Explanations
# =========================
if submitted:
    new_data = pd.DataFrame([{
        "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
        "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob,
        "reason": reason, "guardian": guardian, "traveltime": traveltime,
        "studytime": studytime, "failures": failures, "schoolsup": schoolsup,
        "famsup": famsup, "paid": paid, "activities": activities, "nursery": nursery,
        "higher": higher, "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences, "G1": G1, "G2": G2
    }])

    pred = model.predict(new_data)[0]
    prob = model.predict_proba(new_data)[0][pred]

    # Display result
    if pred == 1:
        st.success(f"✅ Prediction: PASS (Confidence: {prob*100:.1f}%)")
    else:
        st.error(f"❌ Prediction: FAIL (Confidence: {prob*100:.1f}%)")

    # Extract explanation features
    top_feats = get_top_features(model, new_data)

    explanation = generate_explanation(top_feats, pred)
    advice = generate_advice(top_feats)

    full_context = f"""
    Prediction: {'PASS' if pred == 1 else 'FAIL'}
    Confidence: {prob*100:.1f}%

    Top contributing features: {top_feats}

    Reasons:
    {explanation}

    Advice:
    {advice}
    """
    st.subheader("📌 Why this result?")
    st.write(explanation)

    st.subheader("📘 How can the student improve?")
    st.write(advice)
    
    summary_text = generate_summary(pred, top_feats, explanation, advice, prob, new_data)
    st.subheader("📝 Summary (AI-Generated)")
    st.write(summary_text)

