import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("depression.csv")

# Target and features
target_column = "Depression"
cat_features = ["Gender", "City", "Profession"]
num_features = ["Age", "Academic Pressure", "Work Pressure", "CGPA"]

# Train model
def train_model(df, target, cat_features, num_features):
    df_encoded = pd.get_dummies(df[cat_features + num_features])
    X = df_encoded
    y = df[target]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Custom CSS for uniform look
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #faebd7; /* sandal */
    color: black !important;
}

/* All text black */
* {
    color: black !important;
}

/* Input boxes uniform pink */
.stNumberInput input, .stSelectbox div[data-baseweb="select"], 
.stTextInput input, .stTextArea textarea {
    background-color: #ffe4ef !important;
    color: black !important;
    border-radius: 8px;
    border: 1px solid #d3d3d3 !important;
}

/* Dataset preview uniform */
[data-testid="stDataFrame"] {
    background-color: #faebd7 !important;
    color: black !important;
}

/* Buttons */
.stButton>button {
    background-color: #ffe4ef !important;
    color: black !important;
    border-radius: 8px;
    border: 1px solid #d3d3d3 !important;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Depression Prediction App")

# Show dataset
st.subheader("Dataset preview:")
st.dataframe(df)

# User input
st.subheader("Enter your details:")

user_input = {
    "id": st.number_input("id", value=70684.00),
    "Gender": st.selectbox("Gender", df["Gender"].unique().tolist()),
    "Age": st.number_input("Age", value=25.0),
    "City": st.selectbox("City", df["City"].unique().tolist()),
    "Profession": st.selectbox("Profession", df["Profession"].unique().tolist()),
    "Academic Pressure": st.number_input("Academic Pressure", value=3.0),
    "Work Pressure": st.number_input("Work Pressure", value=0.0),
    "CGPA": st.number_input("CGPA", value=7.5),
    "Sleep Duration": st.number_input("Sleep Duration", value=7.0),
    "Dietary Habits": st.selectbox("Dietary Habits", df["Dietary Habits"].unique().tolist() if "Dietary Habits" in df.columns else ["Healthy","Unhealthy"]),
    "Degree": st.text_input("Degree", "B.Pharm"),
    "Suicidal Thoughts": st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"]),
    "Work/Study Hours": st.number_input("Work/Study Hours", value=8.0),
    "Financial Stress": st.number_input("Financial Stress", value=3.0),
    "Family History of Mental Illness": st.selectbox("Family History of Mental Illness", ["Yes", "No"])
}

# Prediction
if st.button("Predict"):
    model = train_model(df, target_column, cat_features, num_features)
    input_df = pd.DataFrame([user_input])

    # Match encoding
    input_encoded = pd.get_dummies(input_df[cat_features + num_features])
    train_encoded = pd.get_dummies(df[cat_features + num_features])
    input_encoded = input_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    pred = model.predict(input_encoded)[0]

    if pred == 1:
        st.markdown("⚠ **You may be experiencing symptoms of depression.**")
        st.markdown("### Suggestions:")
        for msg in [
            "Talk to a trusted friend or family member",
            "Consider speaking with a mental health professional",
            "Practice deep breathing or meditation",
            "Go for a short walk and connect with nature",
            "Remember: You are not alone 💙"
        ]:
            st.markdown(f"- ✅ {msg}")
        st.balloons()
    else:
        st.markdown("🙂 **You do *not* appear to be showing strong signs of depression.**")
        st.markdown("### Keep these up:")
        for msg in [
            "Good sleep, balanced meals, and regular rest help maintain mental health.",
            "Stay connected with friends & family — social support matters.",
            "Engage in hobbies or activities that bring you joy.",
            "Take small breaks when stressed — a short walk helps.",
            "Be mindful of your thoughts; reflection and relaxation matter.",
            "If things change, it’s okay to reach out for help."
        ]:
            st.markdown(f"- ✅ {msg}")
        st.balloons()
