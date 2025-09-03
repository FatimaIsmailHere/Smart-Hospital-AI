import pandas as pd
from sentence_transformers import SentenceTransformer

def embed_notes(input_csv="data/hospital_dataset.csv", output_csv="data/hospital_dataset_embedded.csv"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def embed(text):
        if pd.isna(text):
            return None
        return model.encode(text, normalize_embeddings=True).tolist()
    
    df = pd.read_csv(input_csv)
    df["note_embedding"] = df["ClinicalNotes"].apply(embed)
    df.to_csv(output_csv, index=False)
    print(f"✅ Notes embedded and saved to {output_csv}")


