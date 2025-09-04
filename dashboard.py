import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pymysql, os
from dotenv import load_dotenv
import base64

# --- Import Functions from Your Workflow ---
from scripts.ml import predict_outcome
from scripts.llm_integration import get_llm_recommendations

# Helper function for TiDB connection (copied from your workflow)
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

# --- App Initialization and Layout ---
app = dash.Dash(__name__)
server = app.server

# Paths & Models
MODEL_FILE = Path("models/xgb_model_balanced.pkl")
ENCODER_FILE = Path("models/outcome_label_encoder.pkl")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Visuals and base64 encoding (handle missing files gracefully)
def encode_image(file):
    if not Path(file).exists():
        # Return a simple placeholder in case the file is missing
        placeholder = b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
        return f"data:image/png;base64,{placeholder.decode('utf-8')}"
    with open(file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"

# Styles for a minimalist aesthetic
colors = {
    "background": "#F5F2EC",
    "main-panel": "#FFFFFF",
    "text": "#333333",
    "primary": "#576A5A",
    "border": "#E5E0D4",
    "light-panel": "#F8F5F0",
}

SIDEBAR_STYLE = {
    "padding": "2rem 2rem",
    "backgroundColor": colors["light-panel"],
    "borderRadius": "12px",
    "boxShadow": "0 6px 20px rgba(0,0,0,0.06)",
    "marginRight": "2.5rem",
}

MAIN_CONTENT_STYLE = {
    "padding": "2rem 2rem",
    "backgroundColor": colors["main-panel"],
    "borderRadius": "12px",
    "boxShadow": "0 6px 20px rgba(0,0,0,0.06)",
    "flexGrow": "1",
}

PANEL_TITLE_STYLE = {
    "fontSize": "1.75rem",
    "fontWeight": "700",
    "color": colors["text"],
    "marginBottom": "2rem",
    "borderBottom": f"2px solid {colors['border']}",
    "paddingBottom": "1rem",
}

INPUT_GROUP_STYLE = {
    "marginBottom": "1.5rem",
}

LABEL_STYLE = {
    "display": "block",
    "marginBottom": "0.5rem",
    "color": colors["text"],
    "fontWeight": "600",
    "fontSize": "0.9rem",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "0.75rem 1rem",
    "border": f"1px solid {colors['border']}",
    "borderRadius": "8px",
    "backgroundColor": colors["light-panel"],
    "fontSize": "1rem",
    "color": colors["text"],
}

BUTTON_STYLE = {
    "width": "100%",
    "padding": "0.9rem 1rem",
    "border": "none",
    "borderRadius": "8px",
    "backgroundColor": colors["primary"],
    "color": "white",
    "fontWeight": "bold",
    "fontSize": "1rem",
    "cursor": "pointer",
    "transition": "background-color 0.3s ease",
}

TABLE_HEADER_STYLE = {
    "backgroundColor": colors["border"],
    "fontWeight": "bold",
    "color": colors["text"],
}

app.layout = html.Div(style={"fontFamily": "Inter, sans-serif", "backgroundColor": colors["background"], "padding": "2.5rem 2rem", "color": colors["text"]}, children=[
    html.H1("🏥 Smart Hospital AI Dashboard", style={"textAlign": "center", "fontSize": "3rem", "marginBottom": "3rem", "color": colors["text"], "fontWeight": "700"}),

    html.Div(style={"display": "flex", "maxWidth": "1400px", "margin": "auto"}, children=[
        # Left Panel (Input Form)
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("🩺 Patient Details", style=PANEL_TITLE_STYLE),
            html.Div(style=INPUT_GROUP_STYLE, children=[
                html.Label("Age", style=LABEL_STYLE),
                dcc.Input(id="age", type="number", value=45, min=0, max=100, style=INPUT_STYLE),
            ]),
            html.Div(style=INPUT_GROUP_STYLE, children=[
                html.Label("Gender", style=LABEL_STYLE),
                dcc.Dropdown(id="gender", options=[{"label":"Male","value":"Male"},{"label":"Female","value":"Female"}], value="Male", style=INPUT_STYLE),
            ]),
            html.Div(style=INPUT_GROUP_STYLE, children=[
                html.Label("Diagnoses", style=LABEL_STYLE),
                dcc.Input(id="diagnoses", type="text", value="", style=INPUT_STYLE),
            ]),
            html.Div(style=INPUT_GROUP_STYLE, children=[
                html.Label("Lab Orders", style=LABEL_STYLE),
                dcc.Input(id="lab_orders", type="number", value=5, min=0, max=50, style=INPUT_STYLE),
            ]),
            html.Div(style=INPUT_GROUP_STYLE, children=[
                html.Label("Medications", style=LABEL_STYLE),
                dcc.Input(id="medications", type="text", value="", style=INPUT_STYLE),
            ]),
            html.Div(style=INPUT_GROUP_STYLE, children=[
                html.Label("Readmission Risk", style=LABEL_STYLE),
                dcc.Dropdown(id="readmission_risk",
                    options=[{"label":"Low","value":"Low"},{"label":"Medium","value":"Medium"},{"label":"High","value":"High"}], value="Low", style=INPUT_STYLE),
            ]),
            html.Div(style=INPUT_GROUP_STYLE, children=[
                html.Label("Prior Admissions", style=LABEL_STYLE),
                dcc.Input(id="prior_admissions", type="number", value=1, min=0, max=20, style=INPUT_STYLE),
            ]),
            html.Div(style=INPUT_GROUP_STYLE, children=[
                html.Label("Department", style=LABEL_STYLE),
                dcc.Input(id="department", type="text", value="", style=INPUT_STYLE),
            ]),
            html.Div(style=INPUT_GROUP_STYLE, children=[
                html.Label("Clinical Notes", style=LABEL_STYLE),
                dcc.Textarea(id="clinical_notes", value="", style={**INPUT_STYLE, "height": "100px", "resize": "vertical"}),
            ]),
            html.Button("🔮 Predict Outcome", id="predict_btn", n_clicks=0, style=BUTTON_STYLE),
        ]),

        # Right Panel (Results)
        html.Div(style=MAIN_CONTENT_STYLE, children=[
            html.Div(style={"marginBottom": "2rem"}, children=[
                html.H3("📊 Prediction Result", style=PANEL_TITLE_STYLE),
                html.Div(id="prediction_output", style={"fontSize": "1.1rem", "color": colors["primary"], "fontWeight": "600"}),
            ]),
            html.Div(style={"marginBottom": "2rem"}, children=[
                html.H3("🧬 Similar Patient Cases", style=PANEL_TITLE_STYLE),
                dash_table.DataTable(
                    id="similar_patients_table",
                    style_table={"overflowX": "auto", "border": f"1px solid {colors['border']}", "borderRadius": "8px"},
                    style_header=TABLE_HEADER_STYLE,
                    style_cell={
                        "padding": "12px",
                        "border": "none",
                        "whiteSpace": "normal",
                        "backgroundColor": colors["main-panel"],
                        "color": colors["text"],
                    },
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": colors["light-panel"]}
                    ],
                ),
            ]),
            html.Div(style={"marginBottom": "2rem"}, children=[
                html.H3("💡 AI Recommendations", style=PANEL_TITLE_STYLE),
                html.Div(id="llm_output", style={"fontSize": "1rem", "lineHeight": "1.6", "color": colors["text"]}),
            ]),
            html.Div(children=[
                html.H3("📊 Hospital Insights", style=PANEL_TITLE_STYLE),
                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))", "gap": "1.5rem"}, children=[
                    html.Img(src=encode_image("visuals/age_distribution.png"), style={"width":"100%", "borderRadius":"8px", "boxShadow":"0 2px 5px rgba(0,0,0,0.1)"}),
                    html.Img(src=encode_image("visuals/gender_distribution.png"), style={"width":"100%", "borderRadius":"8px", "boxShadow":"0 2px 5px rgba(0,0,0,0.1)"}),
                    html.Img(src=encode_image("visuals/top_diagnoses.png"), style={"width":"100%", "borderRadius":"8px", "boxShadow":"0 2px 5px rgba(0,0,0,0.1)"}),
                    html.Img(src=encode_image("visuals/correlation_heatmap.png"), style={"width":"100%", "borderRadius":"8px", "boxShadow":"0 2px 5px rgba(0,0,0,0.1)"}),
                ]),
            ]),
        ]),
    ])
])

# --- Callbacks ---
@app.callback(
    Output("prediction_output","children"),
    Output("similar_patients_table","data"),
    Output("similar_patients_table","columns"),
    Output("llm_output","children"),
    Input("predict_btn","n_clicks"),
    State("age","value"),
    State("gender","value"),
    State("diagnoses","value"),
    State("lab_orders","value"),
    State("medications","value"),
    State("readmission_risk","value"),
    State("prior_admissions","value"),
    State("department","value"),
    State("clinical_notes","value"),
)
def update_dashboard(n_clicks, age, gender, diagnoses, lab_orders, medications, readmission_risk,
                     prior_admissions, department, clinical_notes):
    if n_clicks == 0:
        return "", [], [], ""
    
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
    
    note_embedding = EMBED_MODEL.encode(clinical_notes, normalize_embeddings=True).tolist()
    similar_patients = find_similar_patients(note_embedding, top_k=5)
    
    if similar_patients:
        sim_df = pd.DataFrame(similar_patients, columns=["PatientID", "Name", "Diagnoses", "Notes Snip", "Distance"])
        data = sim_df.to_dict("records")
        columns = [{"name":c,"id":c} for c in sim_df.columns]
    else:
        data, columns = [], []
    
    llm_response = get_llm_recommendations(predicted_outcome, patient_row)
    
    return f"Predicted Outcome: {predicted_outcome}", data, columns, llm_response

# --- Main Entry Point ---
if __name__ == "__main__":
    app.run(debug=True)

