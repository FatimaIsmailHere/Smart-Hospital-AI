import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def run_eda(csv_path="data/hospital_dataset.csv"):
    df = pd.read_csv(csv_path)
    df["length_of_stay"] = (pd.to_datetime(df["DischargeDate"]) - pd.to_datetime(df["AdmissionDate"])).dt.days
    bins = [0, 18, 40, 65, 120]
    labels = ["<18", "18-40", "41-65", "65+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    visual_path = Path("visuals")
    visual_path.mkdir(exist_ok=True)

    df["Age"].plot(kind="hist", bins=20, color="skyblue", edgecolor="black")
    plt.title("Age Distribution")
    plt.savefig(visual_path / "age_distribution.png")
    plt.close()

    df["Gender"].value_counts().plot(kind="bar", color="lightgreen")
    plt.title("Gender Distribution")
    plt.savefig(visual_path / "gender_distribution.png")
    plt.close()

    df["Diagnoses"].value_counts().head(10).plot(kind="bar", color="salmon")
    plt.title("Top 10 Diagnoses")
    plt.savefig(visual_path / "top_diagnoses.png")
    plt.close()

    cols = ["Age", "length_of_stay", "PriorAdmissions"]
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(visual_path / "correlation_heatmap.png")
    plt.close()

    summary_file = Path("summary_report.txt")
    with open(summary_file, "w") as f:
        f.write(f"Total patients: {df['PatientID'].nunique()}\n")
        f.write(f"Average age: {round(df['Age'].mean(),1)}\n")
        f.write(f"Gender distribution:\n{df['Gender'].value_counts()}\n")
        f.write(f"Top 5 diagnoses:\n{df['Diagnoses'].value_counts().head(5)}\n")
        f.write(f"Average Length of Stay: {round(df['length_of_stay'].mean(),1)}\n")
    print(f"✅ EDA completed, visuals saved to {visual_path}, summary to {summary_file}")
