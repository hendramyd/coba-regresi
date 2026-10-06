"""Streamlit app for predicting profit with the saved regression model."""

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().with_name("model_regresi_terbaik.pkl")
FEATURE_COLUMNS = ["R&D", "Administrasi", "Marketing", "Wilayah"]
REGIONS = ["New York", "California", "Florida"]


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


st.set_page_config(
    page_title="Prediksi Profit | Model Regresi",
    page_icon="📈",
    layout="wide",
)
st.title("Prediksi Profit")
st.write("Masukkan anggaran operasional dan pilih wilayah untuk memperkirakan profit.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        rd = st.number_input("R&D", min_value=0.0, value=150000.0, step=1000.0)
    with col2:
        administration = st.number_input(
            "Administrasi", min_value=0.0, value=140000.0, step=1000.0
        )
    with col3:
        marketing = st.number_input(
            "Marketing", min_value=0.0, value=300000.0, step=1000.0
        )
    region = st.selectbox("Wilayah", REGIONS)
    submitted = st.form_submit_button("Hitung estimasi profit", type="primary")

if submitted:
    if not MODEL_PATH.is_file():
        st.error(f"File model tidak ditemukan: {MODEL_PATH.name}")
    else:
        input_frame = pd.DataFrame(
            [[rd, administration, marketing, region]], columns=FEATURE_COLUMNS
        )
        try:
            estimate = float(load_model().predict(input_frame)[0])
            st.metric("Estimasi profit", f"${estimate:,.2f}")
        except Exception as error:
            st.error(f"Prediksi gagal: {error}")

st.caption("Hasil merupakan estimasi berdasarkan input dan model yang tersedia.")
