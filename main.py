from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from lime import lime_tabular

app = FastAPI()

# --- 1. Models aur Data Load Karein ---
model_det = joblib.load('det_model.pkl')
model_cls = joblib.load('cls_model.pkl')
scaler_det = joblib.load('scaler_det.pkl')
scaler_cls = joblib.load('scaler_cls.pkl')
cols = joblib.load('feature_names.pkl')
train_sample = joblib.load('training_sample.pkl')

# --- 2. Aapke Actual Features ka Structure ---
class DiagnosisRequest(BaseModel):
    Gender: float
    Age: float
    BSR: float
    Systolic: float
    Diastolic: float
    Peripheral_Neuropathy: float
    BMI: float
    Delayed_Healing: float
    Genetic_Relation: float
    Frequent_Urination: float
    Dry_Mouth: float
    Frequent_Hunger: float
    role: str  # "patient" ya "doctor"

# --- 3. LIME Setup ---
explainer = lime_tabular.LimeTabularExplainer(
    training_data=train_sample,
    feature_names=cols,
    class_names=['Non-Diabetic', 'Diabetic'],
    mode='classification'
)

@app.post("/predict")
def predict_diabetes(request: DiagnosisRequest):
    data_dict = request.dict()
    role = data_dict.pop('role')
    
    # Data ko DataFrame mein convert karna
    input_df = pd.DataFrame([data_dict], columns=cols)
    
    # --- STAGE 1: DETECTION ---
    input_scaled_det = scaler_det.transform(input_df)
    pred_det = model_det.predict(input_scaled_det)[0]
    
    result = {}
    
    # Stage 1 Result
    if pred_det == 0:
        result = {"status": "Non-Diabetic", "stage": 1}
    else:
        # --- STAGE 2: CLASSIFICATION (Diabetic Detected) ---
        input_scaled_cls = scaler_cls.transform(input_df)
        pred_cls = model_cls.predict(input_scaled_cls)[0]
        # Mapping as per  model logic
        type_label = "Diabetic" if pred_cls == 0 else "Gestational"
        result = {"status": "Diabetic Detected", "type": type_label, "stage": 2}

    # --- 4. XAI (Only for Doctor) ---
    if role.lower() == "doctor":
        exp = explainer.explain_instance(
            input_scaled_det[0], 
            model_det.predict_proba, 
            num_features=5
        )
        # Flutter UI ke liye data points
        result["xai_data"] = exp.as_list() 
    
    return result
