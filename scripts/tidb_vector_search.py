import os
import pymysql
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import json

def search_patient(query_text):
    load_dotenv()
    HOST = os.environ["TIDB_HOST"]
    PORT = int(os.environ["TIDB_PORT"])
    USER = os.environ["TIDB_USER"]
    PASS = os.environ["TIDB_PASSWORD"]
    DB   = os.environ["TIDB_DB"]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    qvec = model.encode(query_text, normalize_embeddings=True).tolist()
    qvec_txt = "[" + ",".join(f"{float(x):.7f}" for x in qvec) + "]"

    conn = pymysql.connect(host=HOST, port=PORT, user=USER, password=PASS, database=DB, ssl={"ssl": {}})
    cur = conn.cursor()

    sql = f"""
    SELECT PatientID, Name, Diagnoses, LEFT(ClinicalNotes, 160) AS Snip,
           VEC_COSINE_DISTANCE(note_embedding, %s) AS dist
    FROM patients
    ORDER BY VEC_COSINE_DISTANCE(note_embedding, %s)
    LIMIT 10;
    """

    cur.execute(sql, (qvec_txt, qvec_txt))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

if __name__ == "__main__":
    query_text = input("Enter your query: ")
    results = search_patient(query_text)
    for row in results:
        print(row)
