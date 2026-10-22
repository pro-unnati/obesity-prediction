
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Obesity Risk Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--background-color);
    color: var(--text-color);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Header */
.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(120deg, #102a43, #176b75);
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(16, 42, 67, 0.15);
}

.hero h1 {
    color: white !important;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #e0f2f1 !important;
    font-size: 15px;
    margin-bottom: 0;
}

/* Section headings */
.section-title {
    font-size: 21px;
    font-weight: 750;
    color: var(--text-color);
    margin-top: 10px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: var(--text-color);
    opacity: 0.72;
    font-size: 13px;
    margin-bottom: 18px;
}

/* Information cards */
.info-card {
    background: var(--secondary-background-color);
    color: var(--text-color);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid var(--secondary-background-color);
    box-shadow: 0 4px 14px rgba(16, 42, 67, 0.06);
    min-height: 115px;
}

.info-label {
    color: var(--text-color);
    opacity: 0.7;
    font-size: 13px;
    font-weight: 600;
}

.info-value {
    color: var(--text-color);
    font-size: 26px;
    font-weight: 800;
    margin-top: 8px;
}

/* Result card */
.result-card {
    background: var(--secondary-background-color);
    color: var(--text-color);
    padding: 24px;
    border-radius: 18px;
    border: 1px solid var(--secondary-background-color);
    box-shadow: 0 8px 25px rgba(16, 42, 67, 0.07);
    margin-top: 12px;
}

/* Primary button */
div.stButton > button[kind="primary"],
div.stFormSubmitButton > button[kind="primary"] {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #176b75, #239a91);
    color: white;
    font-weight: 700;
    font-size: 16px;
}

div.stButton > button[kind="primary"]:hover,
div.stFormSubmitButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #125760, #187e78);
    color: white;
    border: none;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--secondary-background-color);
}

/* Input labels and general text */
.stApp label,
.stApp .stMarkdown,
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"] {
    color: var(--text-color);
}

/* Disclaimer */
.disclaimer {
    background: var(--secondary-background-color);
    border-left: 4px solid #e5a000;
    padding: 14px 16px;
    border-radius: 8px;
    color: var(--text-color);
    font-size: 13px;
    margin-top: 20px;
}

/* Keep the sidebar and main page readable in both themes */
[data-testid="stAlert"] {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
MODEL_PATH = Path(__file__).parent / "obesity_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "Model file not found. Please place obesity_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🩺 Obesity Analyzer")
    st.caption("Machine Learning • Health Awareness")

    st.divider()

    st.markdown("### 📌 About this tool")
    st.write(
        "This application estimates an obesity category "
        "using a machine learning model trained on lifestyle "
        "and body-measurement features."
    )

    st.markdown("### 🧾 Model inputs")
    st.write("• Age")
    st.write("• Gender")
    st.write("• Height")
    st.write("• Weight")
    st.write("• Physical activity level")

    st.divider()

    st.caption("Academic ML project — not a medical diagnosis.")


# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
    <h1>🩺 Obesity Risk Analyzer</h1>
    <p>
        Explore how body measurements and physical activity relate
        to predicted obesity categories.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------- TOP INFORMATION CARDS --------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">MODEL TYPE</div>
        <div class="info-value">ML Classification</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">INPUT FEATURES</div>
        <div class="info-value">5 Features</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">BMI</div>
        <div class="info-value">Live Calculation</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# -------------------- INPUT FORM --------------------
st.markdown(
    '<div class="section-title">👤 Personal & Lifestyle Details</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Enter the details below, then select Analyze to view the model prediction.'
    '</div>',
    unsafe_allow_html=True
)

with st.form("obesity_prediction_form"):

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### 🧍 Body Measurements")

        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=25,
            help="Choose the person's age."
        )

        gender = st.selectbox(
            "Gender",
            options=["Female", "Male"],
            help="Select the gender category used by your dataset."
        )

        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=250.0,
            value=165.0,
            step=1.0,
            help="Enter height in centimetres."
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=65.0,
            step=1.0,
            help="Enter weight in kilograms."
        )

    with right:
        st.markdown("#### 🏃 Lifestyle")

        activity_labels = {
            1: "1 — Low / sedentary",
            2: "2 — Light activity",
            3: "3 — Moderate activity",
            4: "4 — High activity"
        }

        activity_level = st.select_slider(
            "Physical Activity Level",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda x: activity_labels[x],
            help=(
                "The numeric activity level must use the same "
                "encoding as the training dataset."
            )
        )

        st.markdown("#### 📊 What happens next?")
        st.info(
            "Your inputs are passed to the trained ML pipeline. "
            "BMI is calculated separately for display and is not "
            "included among the model's five input features."
        )

    submitted = st.form_submit_button(
        "🔍 Analyze My Results",
        type="primary",
        use_container_width=True
    )


# -------------------- BMI CALCULATION --------------------
bmi = weight / ((height / 100) ** 2)

st.write("")

st.markdown(
    '<div class="section-title">📈 Your Measurement Overview</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Age", f"{age} years")

with m2:
    st.metric("Height", f"{height:.0f} cm")

with m3:
    st.metric("Calculated BMI", f"{bmi:.1f} kg/m²")


# General adult BMI screening categories
if bmi < 18.5:
    bmi_category = "Underweight range"
elif bmi < 25:
    bmi_category = "Healthy-weight range"
elif bmi < 30:
    bmi_category = "Overweight range"
else:
    bmi_category = "Obesity range"

st.caption(
    f"General adult BMI screening category: **{bmi_category}**. "
    "BMI is a screening measure, not a diagnosis."
)


# -------------------- PREDICTION --------------------
if submitted:

    if height <= 0 or weight <= 0:
        st.error("Please enter a valid height and weight.")
        st.stop()

    # Keep feature names and order identical to model training.
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Height": height,
        "Weight": weight,
        "PhysicalActivityLevel": activity_level
    }])

    st.divider()

    st.markdown(
        '<div class="section-title">✨ Analysis Result</div>',
        unsafe_allow_html=True
    )

    try:
        prediction = model.predict(input_data)[0]
        prediction_text = str(prediction)

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown("### Predicted Obesity Category")
        st.markdown(f"## {prediction_text}")

        st.write(
            "This is the category predicted by the trained model "
            "for the details you entered."
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Model category scores have been removed.

        st.markdown("""
        <div class="disclaimer">
            <strong>Important:</strong> This is an educational
            machine-learning prediction, not medical advice or
            a clinical diagnosis. Results can be affected by
            dataset limitations and model errors. Consult a
            qualified healthcare professional for personal
            health concerns.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

        st.info(
            "Check that the model was trained using the exact "
            "feature names and category encodings used by this app."
        )


# -------------------- FOOTER --------------------
st.write("")
st.divider()

st.caption(
    "Obesity Risk Analyzer • Built with Python, Pandas and Streamlit • "
    "For educational use"
)
