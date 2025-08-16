import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("brain_weight_predictor.pkl", "rb") as f:
    model = pickle.load(f)

# Page Title
st.set_page_config(page_title="Brain Weight Predictor", page_icon="🧠")
st.title("🧠 Brain Weight Predictor")
st.write("Predict the brain weight based on gender, age, and head size.")

st.markdown("---")  # Separator for better layout

# Input columns
col1, col2 = st.columns(2)

# Gender selection
with col1:
    gender = st.selectbox("Select Gender", ("Male", "Female"))
    gender_val = 1 if gender == "Male" else 2  # Model encoding

# Age selection
with col2:
    age = st.selectbox("Select Age Group", ("Teen", "Adult"))
    age_val = 1 if age == "Teen" else 2  # Model encoding

# Head size input (no upper limit)
head_size = st.number_input(
    "Enter Head Size (cm³)", min_value=50.0, value=1400.0, step=10.0
)

st.markdown("---")

# Predict button
if st.button("Predict Brain Weight"):
    input_data = np.array([[gender_val, age_val, head_size]])
    predicted_weight_array = model.predict(input_data)
    predicted_weight = float(np.squeeze(predicted_weight_array))
    
    # Display result in bigger font
    st.markdown(f"### 🧠 Predicted Brain Weight: {predicted_weight:.2f} grams")

st.info("This prediction is based on a trained model. Accuracy may vary depending on your data.")
