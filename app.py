import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="Depression Prediction App", layout="centered")

# Custom CSS
st.markdown("""
<style>
/* Set background */
.stApp {
    background-color: #fdf6e3;  /* light cream */
}

/* All labels in black */
label, .stMarkdown, .css-10trblm, .css-1d391kg {
    color: #000000 !important;
    font-weight: 600;
}

/* Number input & Selectbox labels */
.stNumberInput label, .stSelectbox label {
    color: #000000 !important;
    font-weight: 600;
}

/* Dropdown text */
.stSelectbox div[data-baseweb="select"] span {
    color: #000000 !important;
}

/* Input text inside fields */
.stNumberInput input, .stTextInput input, .stSelectbox div {
    color: #000000 !important;
    background-color: #ffe6f0 !important; /* light pink */
}

/* Dropdown options panel */
div[data-baseweb="popover"] {
    background-color: #ffe6f0 !important;
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------- APP CONTENT -----------------

st.title("Depression Prediction App")

# Sample dataset preview
data = {
    "id": [2, 8, 26, 30, 32],
    "Gender": ["Male", "Female", "Male", "Female", "Female"],
    "Age": [33, 24, 31, 28, 25],
    "City": ["Visakhapatnam", "Bangalore", "Srinagar", "Varanasi", "Jaipur"],
    "Profession": ["Student"]*5,
    "Academic Pressure": [5, 2, 3, 3, 4],
    "Work Pressure": [0, 0, 0, 0, 0],
    "CGPA": [8.97, 5.9, 7.03, 5.59, 8.13],
}
df = pd.DataFrame(data)
st.subheader("Dataset preview:")
st.dataframe(df)

# User input
st.subheader("Enter your details:")

id_val = st.number_input("id", value=70684.0)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", value=25.0)
city = st.selectbox("City", ["Visakhapatnam", "Bangalore", "Srinagar", "Varanasi", "Jaipur"])
profession = st.selectbox("Profession", ["Student", "Working Professional"])
academic_pressure = st.number_input("Academic Pressure", value=3)
work_pressure = st.number_input("Work Pressure", value=0)
cgpa = st.number_input("CGPA", value=7.5)

# Example prediction button
if st.button("Predict"):
    st.success("Prediction successful! 🎉")
