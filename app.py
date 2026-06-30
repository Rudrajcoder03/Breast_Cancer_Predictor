import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

model = load_model("breast_cancer_model.keras")

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',

    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',

    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

with st.sidebar:

    st.markdown("# 🩺 Breast Cancer Detection")

    st.markdown("---")

    st.markdown("## 📖 About")

    st.write("""
This AI application predicts whether a breast tumor is **Benign** or **Malignant** using a trained Deep Learning model.
""")

    st.markdown("---")

    st.markdown("## 📝 How to Use")

    st.write("""
1. Enter all 30 measurements.

2. Click **Predict**.

3. View the prediction result.
""")

    st.markdown("---")

    st.info(
        "⚠️ This application is for educational purposes only and should not replace medical advice."
    )

st.title("🩺 Breast Cancer Detection System")
st.subheader("AI Powered Deep Learning Model")

st.success(
    "🏥 Early detection saves lives. Enter the 30 tumor measurements below and click **Predict** to get an AI-based diagnosis."
)

st.markdown("---")

st.markdown("## 🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Deep Learning")

with col2:
    st.metric("Features", "30")

with col3:
    st.metric("Framework", "TensorFlow")

st.markdown("---")

st.markdown("---")
st.header("📝 Enter Tumor Measurements")

tabs = st.tabs([
    "🟢 Mean Values",
    "🟠 Error Values",
    "🔴 Worst Values"
])

with tabs[0]:

    col1, col2 = st.columns(2)

    with col1:
        mean_radius = st.number_input("Mean Radius", min_value=0.0)
        mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0)
        mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0)
        mean_concavity = st.number_input("Mean Concavity", min_value=0.0)
        mean_symmetry = st.number_input("Mean Symmetry", min_value=0.0)

    with col2:
        mean_texture = st.number_input("Mean Texture", min_value=0.0)
        mean_area = st.number_input("Mean Area", min_value=0.0)
        mean_compactness = st.number_input("Mean Compactness", min_value=0.0)
        mean_concave_points = st.number_input("Mean Concave Points", min_value=0.0)
        mean_fractal_dimension = st.number_input("Mean Fractal Dimension", min_value=0.0)

with tabs[1]:

    col1, col2 = st.columns(2)

    with col1:
        radius_error = st.number_input("Radius Error", min_value=0.0)
        perimeter_error = st.number_input("Perimeter Error", min_value=0.0)
        smoothness_error = st.number_input("Smoothness Error", min_value=0.0)
        concavity_error = st.number_input("Concavity Error", min_value=0.0)
        symmetry_error = st.number_input("Symmetry Error", min_value=0.0)

    with col2:
        texture_error = st.number_input("Texture Error", min_value=0.0)
        area_error = st.number_input("Area Error", min_value=0.0)
        compactness_error = st.number_input("Compactness Error", min_value=0.0)
        concave_points_error = st.number_input("Concave Points Error", min_value=0.0)
        fractal_dimension_error = st.number_input("Fractal Dimension Error", min_value=0.0)

with tabs[2]:

    col1, col2 = st.columns(2)

    with col1:
        worst_radius = st.number_input("Worst Radius", min_value=0.0)
        worst_perimeter = st.number_input("Worst Perimeter", min_value=0.0)
        worst_smoothness = st.number_input("Worst Smoothness", min_value=0.0)
        worst_concavity = st.number_input("Worst Concavity", min_value=0.0)
        worst_symmetry = st.number_input("Worst Symmetry", min_value=0.0)

    with col2:
        worst_texture = st.number_input("Worst Texture", min_value=0.0)
        worst_area = st.number_input("Worst Area", min_value=0.0)
        worst_compactness = st.number_input("Worst Compactness", min_value=0.0)
        worst_concave_points = st.number_input("Worst Concave Points", min_value=0.0)
        worst_fractal_dimension = st.number_input("Worst Fractal Dimension", min_value=0.0)



input_data = np.array([
    mean_radius,
    mean_texture,
    mean_perimeter,
    mean_area,
    mean_smoothness,
    mean_compactness,
    mean_concavity,
    mean_concave_points,
    mean_symmetry,
    mean_fractal_dimension,

    radius_error,
    texture_error,
    perimeter_error,
    area_error,
    smoothness_error,
    compactness_error,
    concavity_error,
    concave_points_error,
    symmetry_error,
    fractal_dimension_error,

    worst_radius,
    worst_texture,
    worst_perimeter,
    worst_area,
    worst_smoothness,
    worst_compactness,
    worst_concavity,
    worst_concave_points,
    worst_symmetry,
    worst_fractal_dimension
]).reshape(1, -1)
scaled_data = scaler.transform(input_data)

st.info("💡 Please verify that all 30 measurements have been entered before making a prediction.")

st.markdown("---")

predict = st.button(
    "🩺 Predict Breast Cancer",
    use_container_width=True,
    type="primary"
)


if predict:
    scaled_data = scaler.transform(input_data)

prediction = model.predict(scaled_data)

confidence = float(prediction[0][0])

prediction_class = 1 if confidence >= 0.5 else 0

st.markdown("---")
st.header("🩺 Prediction Result")

if prediction_class == 0:

   st.success("✅ Prediction: Benign Tumor")

   st.info("""
   ### What does this mean?

   The AI model predicts that the tumor is **likely benign (non-cancerous)**.

- ✅ Benign tumors generally do not spread to other parts of the body.
- ✅ They are usually less dangerous than malignant tumors.
- 👨‍⚕️ However, please consult a healthcare professional for a proper medical evaluation and confirmation.
""")

   confidence_percent = (1 - confidence) * 100

   st.progress(confidence_percent / 100)

   st.metric(
        "Confidence",
        f"{confidence_percent:.2f}%"
    )

else:

    st.error("⚠️ Prediction: Malignant Tumor")

    st.warning("""
    ### What does this mean?

The AI model predicts that the tumor is **likely malignant (cancerous)**.

- 🔴 Malignant tumors can invade nearby tissues.
- 🔴 They may spread to other parts of the body (metastasis) if left untreated.
- 🚨 Early diagnosis and treatment greatly improve outcomes.
- 👨‍⚕️ Please consult an oncologist or healthcare professional as soon as possible for further tests and medical advice.

**Note:** This AI prediction is **not a medical diagnosis** and should not replace professional medical evaluation.
""")

    confidence_percent = confidence * 100

    st.progress(confidence_percent / 100)

    st.metric(
        "Confidence",
        f"{confidence_percent:.2f}%"
    )

st.info("""
### 💡 Recommendation

The model predicts the tumor is **Benign**.

✔ Regular health checkups are still recommended.

✔ Always consult a healthcare professional for confirmation.
""")

st.warning("""
### ⚠ Recommendation

The model predicts the tumor is **Malignant**.

Please consult an oncologist immediately.

This prediction is **not a medical diagnosis**.
""")

st.markdown("---")

st.warning("""
### ⚠️ Medical Disclaimer

This application is intended for educational purposes only.

The prediction generated by this AI model should **not** be considered a medical diagnosis.

Please consult a qualified healthcare professional for medical advice and diagnosis.
""")

st.markdown("---")

st.markdown(
"""
<div style='text-align:center; color:gray;'>

### 👨‍💻 Developed by Rudraj Kumar Nath

AI/ML Intern @ InternPe

Built using TensorFlow • Streamlit • Python

</div>
""",
unsafe_allow_html=True
)






