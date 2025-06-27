import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and data
model = joblib.load("disease_treatment_model.pkl")
df = pd.read_csv("disease_treatment_500.csv")
disease_list = sorted(df["Disease"].str.title().unique())

st.set_page_config(page_title="Disease Treatment Recommender", page_icon="💊")
st.title("💊 Advanced Treatment Recommendation System")
st.markdown("Select or type a disease name below to get the **top 3 recommended treatments** with confidence scores.")

# Input with Autocomplete
disease_input = st.selectbox("Select or Type Disease Name:", [""] + disease_list)

if st.button("Get Treatment Recommendations"):
    if disease_input.strip():
        disease_text = disease_input.strip().lower()
        # Predict probabilities
        try:
            proba = model.predict_proba([disease_text])[0]
            classes = model.classes_
            top_indices = np.argsort(proba)[::-1][:3]
            st.success(f"🦠 **Disease:** {disease_input}")
            st.subheader("💊 **Top 3 Recommended Treatments:**")
            for idx in top_indices:
                treatment = classes[idx]
                confidence = proba[idx] * 100
                st.markdown(f"- **Treatment:** {treatment}\n  - Confidence: **{confidence:.2f}%**")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.warning("⚠️ Please select or type a disease name.")
