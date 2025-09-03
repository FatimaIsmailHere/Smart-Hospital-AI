import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scripts.ml import predict_outcome
from scripts.llm_integration import get_llm_recommendations
from scripts.tidb_module import get_connection, init_table, ingest_csv
import pymysql, os
from dotenv import load_dotenv

# ----------------------------
# Paths
# ----------------------------
MODEL_FILE = Path("models/xgb_model_balanced.pkl")
ENCODER_FILE = Path("models/outcome_label_encoder.pkl")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Helper: Find Similar Patients
# ----------------------------
def find_similar_patients(note_embedding, top_k=5):
    load_dotenv()
    HOST = os.environ["TIDB_HOST"]
    PORT = int(os.environ["TIDB_PORT"])
    USER = os.environ["TIDB_USER"]
    PASS = os.environ["TIDB_PASSWORD"]
    DB   = os.environ["TIDB_DB"]

    qvec_txt = "[" + ",".join(f"{float(x):.7f}" for x in note_embedding) + "]"

    conn = pymysql.connect(
        host=HOST, port=PORT, user=USER, password=PASS,
        database=DB, ssl={"ssl": {}}
    )
    cur = conn.cursor()
    sql = f"""
    SELECT PatientID, Name, Diagnoses, LEFT(ClinicalNotes, 160) AS Snip,
           VEC_COSINE_DISTANCE(note_embedding, %s) AS dist
    FROM patients
    ORDER BY VEC_COSINE_DISTANCE(note_embedding, %s)
    LIMIT {top_k};
    """
    cur.execute(sql, (qvec_txt, qvec_txt))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🏥 Smart Hospital Dashboard", layout="wide")
st.title("🏥 Smart Hospital AI Dashboard")

# Sidebar Patient Input Form
st.sidebar.header("🩺 Enter Patient Details")
age = st.sidebar.number_input("Age", 0, 100, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
diagnoses = st.sidebar.text_input("Diagnoses")
lab_orders = st.sidebar.number_input("Lab Orders", 0, 50, 5)
medications = st.sidebar.text_input("Medications")
readmission_risk = st.sidebar.selectbox("Readmission Risk", ["Low", "Medium", "High"])
prior_admissions = st.sidebar.number_input("Prior Admissions", 0, 20, 1)
department = st.sidebar.text_input("Department")
clinical_notes = st.sidebar.text_area("Clinical Notes", height=120)

if st.sidebar.button("🔮 Predict Outcome"):
    # ----------------------------
    # Step 1: ML Prediction
    # ----------------------------
    patient_row = {
        "Age": age,
        "Gender": gender,
        "Diagnoses": diagnoses,
        "LabOrders": lab_orders,
        "Medications": medications,
        "ReadmissionRisk": readmission_risk,
        "PriorAdmissions": prior_admissions,
        "Department": department
    }
    predicted_outcome = predict_outcome(str(MODEL_FILE), str(ENCODER_FILE), patient_row)

    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Outcome: **{predicted_outcome}**")

    # ----------------------------
    # Step 2: Similar Patients
    # ----------------------------
    st.subheader("🧬 Similar Patient Cases")
    note_embedding = EMBED_MODEL.encode(clinical_notes, normalize_embeddings=True).tolist()
    similar_patients = find_similar_patients(note_embedding, top_k=5)

    if similar_patients:
        sim_df = pd.DataFrame(similar_patients, columns=["PatientID", "Name", "Diagnoses", "Notes Snip", "Distance"])
        st.dataframe(sim_df, use_container_width=True)
    else:
        st.warning("No similar patients found in DB.")

    # ----------------------------
    # Step 3: LLM Recommendations
    # ----------------------------
    st.subheader("💡 AI Recommendations")
    llm_response = get_llm_recommendations(predicted_outcome, patient_row)
    st.info(llm_response)
    # ----------------------------
# Step 4: Hospital Insights (EDA Visuals)
# ----------------------------
st.subheader("📊 Hospital Insights")

col1, col2 = st.columns(2)
with col1:
    st.image("visuals/age_distribution.png", caption="Age Distribution", use_container_width=True)
    st.image("visuals/gender_distribution.png", caption="Gender Distribution", use_container_width=True)

with col2:
    st.image("visuals/top_diagnoses.png", caption="Top 10 Diagnoses", use_container_width=True)
    st.image("visuals/correlation_heatmap.png", caption="Correlation Heatmap", use_container_width=True)

# Add summary report if available
summary_path = Path("summary_report.txt")
if summary_path.exists():
    st.subheader("📄 Summary Report")
    with open(summary_path, "r") as f:
        st.text(f.read())

