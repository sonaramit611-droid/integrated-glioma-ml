
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Integrated Glioma Predictor",
    page_icon="🧠",
    layout="wide"
)

# Load models
grade_model = joblib.load("glioma_high_grade_model.joblib")
idh_model = joblib.load("glioma_idh_model.joblib")

st.title("🧠 Integrated Glioma ML Predictor")
st.write("Clinical + MRI based prediction of high-grade glioma and IDH mutation.")

st.header("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=1, max_value=100, value=50)
    Sex = st.selectbox("Sex", ["Male", "Female"])
    Headache = st.selectbox("Headache", [0, 1])
    Seizure = st.selectbox("Seizure", [0, 1])
    Focal_Deficit = st.selectbox("Focal Deficit", [0, 1])
    KPS = st.number_input("KPS", min_value=0, max_value=100, value=80)

with col2:
    Tumor_Location = st.selectbox(
        "Tumor Location",
        ["Frontal", "Temporal", "Parietal", "Occipital", "Other"]
    )
    Tumor_Side = st.selectbox("Tumor Side", ["Left", "Right", "Midline"])
    Tumor_Size_cm = st.number_input("Tumor Size (cm)", min_value=0.0, value=4.0)
    Tumor_Volume_cm3 = st.number_input("Tumor Volume (cm³)", min_value=0.0, value=30.0)
    Edema = st.selectbox("Edema", [0, 1])
    Necrosis = st.selectbox("Necrosis", [0, 1])
    Hemorrhage = st.selectbox("Hemorrhage", [0, 1])

with col3:
    T1_Hypointense = st.selectbox("T1 Hypointense", [0, 1])
    T2_Hyperintense = st.selectbox("T2 Hyperintense", [0, 1])
    Contrast_Enhancement = st.selectbox("Contrast Enhancement", [0, 1])
    Enhancement_Pattern = st.selectbox(
        "Enhancement Pattern",
        ["None", "Homogeneous", "Heterogeneous", "Ring"]
    )
    Restricted_Diffusion = st.selectbox("Restricted Diffusion", [0, 1])
    ADC_Low = st.selectbox("Low ADC", [0, 1])
    Midline_Shift_mm = st.number_input("Midline Shift (mm)", min_value=0.0, value=0.0)
    Multifocal = st.selectbox("Multifocal", [0, 1])

if st.button("🔍 Predict"):

    patient = pd.DataFrame([{
        "Age": Age,
        "Sex": Sex,
        "Headache": Headache,
        "Seizure": Seizure,
        "Focal_Deficit": Focal_Deficit,
        "KPS": KPS,
        "Tumor_Location": Tumor_Location,
        "Tumor_Side": Tumor_Side,
        "Tumor_Size_cm": Tumor_Size_cm,
        "Tumor_Volume_cm3": Tumor_Volume_cm3,
        "Edema": Edema,
        "Necrosis": Necrosis,
        "Hemorrhage": Hemorrhage,
        "T1_Hypointense": T1_Hypointense,
        "T2_Hyperintense": T2_Hyperintense,
        "Contrast_Enhancement": Contrast_Enhancement,
        "Enhancement_Pattern": Enhancement_Pattern,
        "Restricted_Diffusion": Restricted_Diffusion,
        "ADC_Low": ADC_Low,
        "Midline_Shift_mm": Midline_Shift_mm,
        "Multifocal": Multifocal
    }])

    grade_probability = grade_model.predict_proba(patient)[0, 1] * 100
    idh_probability = idh_model.predict_proba(patient)[0, 1] * 100

    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "High-grade Glioma (WHO 3–4)",
            f"{grade_probability:.1f}%"
        )

    with col2:
        st.metric(
            "IDH Mutation",
            f"{idh_probability:.1f}%"
        )

    st.info(
        "This is a research/demo prediction model and must not be used "
        "as a substitute for histopathology or molecular testing."
    )
