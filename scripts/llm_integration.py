import subprocess

def get_llm_recommendations(predicted_outcome, patient_info):
    prompt = f"""
The ML model predicted the patient's outcome as: {predicted_outcome}.

Patient details: {patient_info}

👉 Based on this, provide:
1. A short explanation of why this outcome might happen.
2. 3 practical recommendations for follow-up care or monitoring.
"""
    result = subprocess.run(
        ["ollama", "run", "llama3:latest"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8")

