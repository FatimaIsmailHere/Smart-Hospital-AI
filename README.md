
# 🏥 Smart Hospital AI

> **Unique 4-in-1 Hospital Intelligence Dashboard**
> Predict patient risk, retrieve similar cases, generate AI recommendations, and run hospital-wide surge simulations.

---

## 🚀 Features

* **📊 Prediction Result** – Classifies patients into *High, Medium, Low risk*.
* **🧬 Similar Patient Cases** – Retrieve cases using **TiDB Vector Search**.
* **💡 AI Recommendations** – Actionable, AI-powered suggestions for care & staffing.
* **📈 Hospital Insights** – Surge simulations and resource allocation dashboards.

---

## 🔄 Workflow

1. **Data Input** → Patient details entered into the system.
2. **Risk Prediction** → XGBoost model classifies patient.
3. **Case Retrieval** → SentenceTransformers + TiDB Vector Search fetch similar cases.
4. **AI Recommendations** → LLM generates guidance.
5. **Insights Dashboard** → Dash + Plotly visualize hospital-wide trends.

---

## 🛠️ Built With

Python, XGBoost, SentenceTransformers, TiDB Serverless, TiDB Vector Search, OpenAI, Dash, Plotly, Pandas, NumPy, Joblib, dotenv, PyMySQL

---

## ⚡ Installation & Running

### 1️⃣ Clone the repo

```bash
git clone https://github.com/FatimaIsmailHere/Smart-Hospital-AI.git
cd Smart-Hospital-AI
```

### 2️⃣ Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate   # (Windows PowerShell)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup environment variables

Create a **`.env`** file in the root folder:

```
TIDB_HOST=your_tidb_host
TIDB_USER=your_tidb_user
TIDB_PASSWORD=your_tidb_password
OPENAI_API_KEY=your_openai_key
```

### 5️⃣ Run the app

```bash
python dashboard.py
```

App will be available at → **[http://127.0.0.1:8050/](http://127.0.0.1:8050/)**

---

## 📊 Dataset
* A sample dataset (`hospital_dataset.xlsx`) is provided in the `data/` folder.
* For larger datasets, update `data/` and adjust `scripts/ml.py`.

---

## 🧪 Evaluation

* Tested on edge cases & unusual patient profiles.
* Simulated ER surges to generate staffing and intervention recommendations.
* Documented limitations and possible improvements.

---

## 🌍 Vision & Next Steps
* Clinical validation with real hospital data.
* Time-series forecasting for ER surges (24–72 hours ahead).
* IoT/wearable integration for real-time monitoring.
* Deploy as **Hospital Command Center AI** for global healthcare optimization.

---

## 📜 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) file for details.

---

