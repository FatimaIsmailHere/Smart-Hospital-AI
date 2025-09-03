import os
from pathlib import Path
import pandas as pd
from scripts.make_embeddings import embed_notes
from scripts.tidb_module import get_connection, init_table, ingest_csv
from scripts.ml import train_and_save_models, predict_outcome
from scripts.llm_integration import get_llm_recommendations
from sentence_transformers import SentenceTransformer

# ----------------------------
# Paths
# ----------------------------
RAW_CSV = Path("data/hospital_dataset.csv")
EMBED_CSV = Path("data/hospital_dataset_embedded.csv")
MODEL_FILE = Path("models/xgb_model_balanced.pkl")
ENCODER_FILE = Path("models/outcome_label_encoder.pkl")

# ----------------------------
# 1️⃣ Embed Clinical Notes (skip if exists)
# ----------------------------
if not EMBED_CSV.exists():
    print("\n=== Step 1: Embedding Clinical Notes ===")
    embed_notes(input_csv=RAW_CSV, output_csv=EMBED_CSV)
else:
    print(f"\n✅ Step 1 skipped: Embedded CSV already exists at {EMBED_CSV}")

# ----------------------------
# 2️⃣ Ingest into TiDB (skip if table exists)
# ----------------------------
if not EMBED_CSV.exists():  # Only ingest if embedding just created
    conn = get_connection()
    init_table(conn)
    ingest_csv(conn, csv_path=EMBED_CSV)
    conn.close()
    print("\n✅ Data ingested into TiDB")
else:
    print("\n✅ Step 2 skipped: TiDB ingestion not needed")

# ----------------------------
# 3️⃣ Train ML Models (skip if already trained)
# ----------------------------
if not MODEL_FILE.exists() or not ENCODER_FILE.exists():
    print("\n=== Step 3: Training ML Models ===")
    train_and_save_models(csv_path=EMBED_CSV)
else:
    print("\n✅ Step 3 skipped: Models already trained")

# ----------------------------
# 4️⃣ ML Prediction
# ----------------------------
print("\n=== Step 4: ML Prediction ===")
print("Please enter patient details:")

age = int(input("Age: "))
gender = input("Gender: ")
diagnoses = input("Diagnoses: ")
lab_orders = int(input("Lab Orders: "))
medications = input("Medications: ")
readmission_risk = input("Readmission Risk: ")
prior_admissions = int(input("Prior Admissions: "))
department = input("Department: ")
clinical_notes = input("Clinical Notes (detailed patient notes): ")

# ✅ Keep embedding only for semantic search (not ML model)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
note_embedding = embed_model.encode(clinical_notes, normalize_embeddings=True).tolist()

# Prepare DataFrame (without embeddings for ML model)
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


# Predict outcome
predicted_outcome = predict_outcome(str(MODEL_FILE), str(ENCODER_FILE),patient_row)
print(f"\nPredicted outcome: {predicted_outcome}")

# ----------------------------
# 5️⃣ Find Similar Patients
# ----------------------------
import pymysql
from dotenv import load_dotenv
import os

def find_similar_patients(note_embedding, top_k=5):
    load_dotenv()
    HOST = os.environ["TIDB_HOST"]
    PORT = int(os.environ["TIDB_PORT"])
    USER = os.environ["TIDB_USER"]
    PASS = os.environ["TIDB_PASSWORD"]
    DB   = os.environ["TIDB_DB"]

    # Convert embedding to SQL text format
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

print("\n=== Step 5: Similar Patients ===")
similar_patients = find_similar_patients(note_embedding, top_k=5)
for p in similar_patients:
    print(p)

# ----------------------------
# 6️⃣ LLM Recommendations
# ----------------------------
llm_response = get_llm_recommendations(predicted_outcome, patient_row)
print("\n💡 LLM Recommendations:\n", llm_response)

print("\n✅ Workflow completed successfully!")


print("\n✅ Workflow completed successfully!")
