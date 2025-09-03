import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import joblib

def train_and_save_models(csv_path="data/hospital_dataset_embedded.csv"):
    df = pd.read_csv(csv_path)

    # Drop columns not useful for ML model
    drop_cols = [
        "PatientID", "Name", "Outcome",
        "ClinicalNotes", "AdmissionDate", "DischargeDate",
        "note_embedding" 
    ]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Outcome"]

    # Encode categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))

    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Handle class imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    sample_weights = np.array([dict(zip(classes, weights))[y] for y in y_train])

    # Train XGBoost
    xgb_balanced = XGBClassifier(
        objective='multi:softmax',
        num_class=len(classes),
        eval_metric='mlogloss',
        random_state=42
    )
    xgb_balanced.fit(X_train, y_train, sample_weight=sample_weights)

    # Save model + encoder
    joblib.dump(xgb_balanced, "models/xgb_model_balanced.pkl")
    joblib.dump(le_target, "models/outcome_label_encoder.pkl")
    print("✅ ML model trained and saved")
    return X_test, y_test, le_target, xgb_balanced


# ================= Prediction Function ================= #
import joblib
from sklearn.preprocessing import LabelEncoder

def predict_outcome(model_path, encoder_path, patient_row):
    # Load trained model & encoder
    model = joblib.load(model_path)
    le_target = joblib.load(encoder_path)

    # Convert patient_row to DataFrame
    df = pd.DataFrame([patient_row])


    # Encode categorical features (⚠️ Ideally use saved encoders for consistency)
    cat_cols = ["Gender", "Diagnoses", "LabOrders", "Medications", "ReadmissionRisk", "Department"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Predict
    pred_encoded = model.predict(df)[0]
    pred_label = le_target.inverse_transform([pred_encoded])[0]
    return pred_label

