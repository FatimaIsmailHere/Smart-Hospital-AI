import os
import pandas as pd
import pymysql
import json
from dotenv import load_dotenv

def get_connection():
    load_dotenv()
    conn = pymysql.connect(
        host=os.environ["TIDB_HOST"],
        port=int(os.environ["TIDB_PORT"]),
        user=os.environ["TIDB_USER"],
        password=os.environ["TIDB_PASSWORD"],
        database=os.environ["TIDB_DB"],
        ssl={"ssl": {}}
    )
    return conn

def init_table(conn):
    cur = conn.cursor()
    table = "patients"
    db = os.environ["TIDB_DB"]
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db}`;")
    cur.execute(f"USE `{db}`;")
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS `{table}` (
        PatientID INT PRIMARY KEY,
        Name VARCHAR(100),
        Age INT,
        Gender VARCHAR(10),
        Diagnoses TEXT,
        LabOrders TEXT,
        Medications TEXT,
        ReadmissionRisk VARCHAR(20),
        PriorAdmissions INT,
        ClinicalNotes TEXT,
        AdmissionDate DATE,
        DischargeDate DATE,
        Department VARCHAR(50),
        Outcome VARCHAR(20),
        note_embedding VECTOR(384)
    );
    """)
    cur.execute(f"ALTER TABLE {table} SET TIFLASH REPLICA 1;")
    conn.commit()
    cur.close()
    print(f"✅ Table `{table}` initialized")

def ingest_csv(conn, csv_path="data/hospital_dataset_embedded.csv"):
    cur = conn.cursor()
    df = pd.read_csv(csv_path)
    df["AdmissionDate"] = pd.to_datetime(df["AdmissionDate"], errors='coerce').dt.strftime('%Y-%m-%d')
    df["DischargeDate"] = pd.to_datetime(df["DischargeDate"], errors='coerce').dt.strftime('%Y-%m-%d')

    def to_vec_text(lst):
        if isinstance(lst, str):
            lst = json.loads(lst)
        return "[" + ",".join(f"{float(x):.7f}" for x in lst) + "]"

    rows = []
    for _, r in df.iterrows():
        rows.append((
            int(r["PatientID"]),
            str(r["Name"]),
            int(r["Age"]),
            str(r["Gender"]),
            str(r["Diagnoses"]),
            str(r["LabOrders"]),
            str(r["Medications"]),
            str(r["ReadmissionRisk"]),
            int(r["PriorAdmissions"]),
            str(r["ClinicalNotes"]),
            str(r["AdmissionDate"]),
            str(r["DischargeDate"]),
            str(r["Department"]),
            str(r["Outcome"]),
            to_vec_text(r["note_embedding"])
        ))

    sql = f"""
    INSERT INTO patients
    (PatientID,Name,Age,Gender,Diagnoses,LabOrders,Medications,ReadmissionRisk,
     PriorAdmissions,ClinicalNotes,AdmissionDate,DischargeDate,Department,Outcome,
     note_embedding)
    VALUES
    (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,CAST(%s AS VECTOR));
    """
    BATCH = 500
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i+BATCH]
        cur.executemany(sql, chunk)
        conn.commit()
    cur.close()
    print("✅ Data ingested into TiDB")
