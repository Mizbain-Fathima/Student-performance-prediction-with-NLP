# student_pass_fail_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1) Load dataset ------------------------------------------------------------
df = pd.read_csv("student-por.csv")

# 2) Create Pass/Fail column (1 = Pass, 0 = Fail)
df["Pass"] = (df["G3"] >= 10).astype(int)

# Target & features
y = df["Pass"]
X = df.drop(columns=["G3", "Pass"])

# 3) Identify categorical & numeric features --------------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# 4) Preprocessor ------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# 5) Model pipeline ----------------------------------------------------------
clf = DecisionTreeClassifier(random_state=42)
pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])

# 6) Train-test split --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7) Hyperparameter tuning ---------------------------------------------------
param_grid = {
    "model__max_depth": [4, 6, 8, 10, 12],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}

grid = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("\nBest Params:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# 8) Evaluate ----------------------------------------------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

def get_top_features(model, input_df):
    # Get feature importances (for tree models)
    importances = model.feature_importances_
    feature_names = input_df.columns

    # Pair them
    pairs = list(zip(feature_names, importances))
    # Sort highest → lowest
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    
    # Return top 3 features
    return [feat for feat, score in sorted_pairs[:3]]

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

reason_map = {
    'G1': "Your first period grade (G1) was low, indicating weak initial understanding.",
    'G2': "Your second period grade (G2) was low, which strongly affects the final result.",
    'absences': "You have a high number of absences, reducing continuity in learning.",
    'failures': "You previously failed subjects, suggesting foundational gaps.",
    'studytime': "Your weekly study time is low, which impacts preparation.",
    'freetime': "Excess free time compared to study time affects performance.",
    'goout': "Going out frequently can reduce study hours and focus."
}

advice_map = {
    'G1': "Revise core topics from earlier exams and solve past papers.",
    'G2': "Improve continuous assessment performance through weekly revision.",
    'absences': "Reduce absences — aim for at least 90% attendance.",
    'failures': "Focus on strengthening basics from failed subjects with targeted practice.",
    'studytime': "Increase study time to at least 2 hours/day using Pomodoro technique.",
    'freetime': "Balance free time and studies by maintaining a structured daily routine.",
    'goout': "Limit outings on weekdays to stay consistent with studies."
}

def generate_explanation(top_features, reason_map, prediction_label):
    reasons = [reason_map.get(f, "") for f in top_features]
    reasons = [r for r in reasons if r]  # remove empty
    
    if prediction_label == "Pass":
        text = "You are likely to PASS. Key positive factors:\n- " + "\n- ".join(reasons)
    else:
        text = "You are likely to FAIL. Key contributing reasons:\n- " + "\n- ".join(reasons)

    return text

def generate_advice(top_features, advice_map):
    tips = [advice_map.get(f, "") for f in top_features]
    tips = [t for t in tips if t]

    if not tips:
        return "Maintain consistent study habits and focus on continuous improvement."

    return "To improve further:\n- " + "\n- ".join(tips)

# 9) Save model --------------------------------------------------------------
joblib.dump(best_model, "best_pass_fail_model.joblib", compress=3)
print("\nSaved: best_pass_fail_model.joblib")
