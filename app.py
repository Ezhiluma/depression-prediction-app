import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- Custom CSS ---
st.markdown("""
<style>
/* Page background */
.stApp {
    background-color: #f5e6cc;  /* sandal / light peach */
    font-family: 'Segoe UI', sans-serif;
    color: #444444; /* mild grey for general text */
}

/* Main title (colored) */
h1 {
    color: #4a6fa5 !important; /* soft blue */
    font-weight: 700 !important;
}

/* Subheaders / Topics */
h2, h3, h4, h5, h6 {
    color: #000000 !important; /* black */
    font-weight: 700 !important;
}

/* Labels (questions under Enter your details) */
label, .stNumberInput label, .stSelectbox label {
    display: block !important;
    font-weight: 700 !important;
    color: #000000 !important;  /* black */
    margin-bottom: 6px !important;
    background: transparent !important;
    padding: 0 !important;
    border-radius: 0 !important;
}

/* Input boxes */
.stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border: 1px solid #cfd8dc !important;
    border-radius: 6px !important;
    padding: 6px !important;
    color: #333333 !important; /* grey text */
    font-weight: 500 !important;
}

/* Dropdown menu options */
div[data-baseweb="popover"] {
    background-color: #ffffff !important;
    color: #333333 !important;
}

/* Prediction messages box */
.stAlert {
    background-color: #ffe6f0 !important; /* light pink */
    color: #333333 !important;
    border-radius: 12px;
    padding: 12px;
    font-weight: 600;
}

/* Dataset preview (black background) */
.stDataFrame div {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    padding: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# --- Cache dataset loading ---
@st.cache_data
def load_default_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

# --- Cache model training ---
@st.cache_data
def train_model(df, target_column, cat_features, num_features):
    X = df[cat_features + num_features]
    y = df[target_column]

    cat_transformer = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("cat", cat_transformer, cat_features),
        ("num", num_transformer, num_features)
    ])

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier())
    ])
    clf.fit(X, y)
    return clf

# --- Main Streamlit App ---
st.set_page_config(page_title="Depression Predictor", layout="centered")
st.title("Depression Prediction App")

data_path = "depression.csv"
if not os.path.exists(data_path):
    st.error(f"Dataset {data_path} not found in app folder.")
    st.stop()

df = load_default_data(data_path)
st.subheader("Dataset preview:")
st.write(df.head())

target_column = "Depression"
if target_column not in df.columns:
    st.error(f"Target column '{target_column}' not found.")
    st.stop()

# Features
feature_cols = [c for c in df.columns if c != target_column]
cat_features = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
num_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

st.subheader("Enter your details:")

# User input
user_input = {}
for c in feature_cols:
    if c in num_features:
        col_series = df[c].dropna()
        min_val = float(col_series.min())
        max_val = float(col_series.max())
        default_val = float(col_series.median())
        step_val = 0.1 if (max_val - min_val) < 1 else 1.0
        user_input[c] = st.number_input(
            f"{c}",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step_val,
            format="%.2f"
        )
    else:
        options = df[c].dropna().unique().tolist()
        user_input[c] = st.selectbox(f"{c}", options)

# Predict
if st.button("Predict"):
    model = train_model(df, target_column, cat_features, num_features)
    input_df = pd.DataFrame([user_input])
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("⚠ You may be experiencing symptoms of depression.")
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
        st.success("🙂 You do *not* appear to be showing strong signs of depression.")
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
