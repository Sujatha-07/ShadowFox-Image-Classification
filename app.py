
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ShadowFox | AI Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        color: #172554;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: #1e293b;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #94a3b8;
        border-radius: 14px;
        padding: 1rem;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 12px;
    }

    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
        border: none;
    }

    [data-testid="stProgress"] > div > div {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CLASS_NAMES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

MODEL_PATH = Path("models/image_classifier_transfer.keras")
IMG_SIZE = (96, 96)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.info(
        "Please check whether the trained model exists "
        "inside the models folder."
    )
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔍 AI Image Classification")

st.write(
    "Upload one or more images and classify them using "
    "a trained MobileNetV2 deep learning model."
)

st.divider()

# --------------------------------------------------
# MULTIPLE IMAGE UPLOAD
# --------------------------------------------------

st.subheader("📤 Upload Your Images")

uploaded_files = st.file_uploader(
    "Choose JPG or PNG images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="You can upload multiple images using the + option.",
    key="image_uploader",
)

# --------------------------------------------------
# INITIAL MESSAGE
# --------------------------------------------------

if not uploaded_files:
    st.info(
        "👆 Upload one or more images to begin classification."
    )

# --------------------------------------------------
# DISPLAY UPLOADED IMAGES
# --------------------------------------------------

else:
    st.success(
        f"Successfully uploaded {len(uploaded_files)} image(s)."
    )

    with st.expander("View Uploaded Images"):
        preview_columns = st.columns(3)

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                preview_image = Image.open(
                    uploaded_file
                ).convert("RGB")

                with preview_columns[index % 3]:
                    st.image(
                        preview_image,
                        caption=uploaded_file.name,
                        use_container_width=True,
                    )

            except Exception:
                st.warning(
                    f"Could not preview {uploaded_file.name}"
                )

    st.divider()

    # --------------------------------------------------
    # CLASSIFY ALL IMAGES
    # --------------------------------------------------

    classify_all = st.button(
        "🚀 Classify All Images",
        type="primary",
        use_container_width=True,
        key="classify_all",
    )

    if classify_all:
        results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.write(
                    f"Analyzing image {index + 1} of "
                    f"{len(uploaded_files)}: "
                    f"{uploaded_file.name}"
                )

                # Read image
                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                # Resize image
                resized_image = image.resize(IMG_SIZE)

                # Convert image to NumPy array
                image_array = np.array(
                    resized_image,
                    dtype=np.float32,
                )

                # MobileNetV2 preprocessing
                image_array = (
                    tf.keras.applications.mobilenet_v2
                    .preprocess_input(image_array)
                )

                # Add batch dimension
                image_array = np.expand_dims(
                    image_array,
                    axis=0,
                )

                # Predict
                probabilities = model.predict(
                    image_array,
                    verbose=0,
                )[0]

                # Find predicted class
                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = CLASS_NAMES[
                    predicted_index
                ]

                confidence = float(
                    probabilities[predicted_index] * 100
                )

                # Find top 3 predictions
                top_indices = np.argsort(
                    probabilities
                )[::-1][:3]

                top_predictions = []

                for class_index in top_indices:
                    top_predictions.append({
                        "class": CLASS_NAMES[int(class_index)],
                        "confidence": float(
                            probabilities[class_index] * 100
                        ),
                    })

                results.append({
                    "filename": uploaded_file.name,
                    "image": image.copy(),
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "top_predictions": top_predictions,
                    "error": None,
                })

            except Exception as error:
                results.append({
                    "filename": uploaded_file.name,
                    "image": None,
                    "predicted_class": None,
                    "confidence": 0.0,
                    "top_predictions": [],
                    "error": str(error),
                })

            # Update progress
            progress_bar.progress(
                (index + 1) / len(uploaded_files)
            )

        # Save results in session state
        st.session_state["classification_results"] = results

        st.session_state["classified_filenames"] = [
            (file.name, file.size)
            for file in uploaded_files
        ]

        status_text.success(
            "✅ Classification completed successfully!"
        )

# --------------------------------------------------
# DISPLAY CLASSIFICATION RESULTS
# --------------------------------------------------

results = st.session_state.get(
    "classification_results", []
)

current_filenames = [
    (file.name, file.size)
    for file in uploaded_files
] if uploaded_files else []

previous_filenames = st.session_state.get(
    "classified_filenames", []
)

if results and current_filenames == previous_filenames:

    st.divider()
    st.subheader("📊 Prediction Results")

    successful_results = [
        result for result in results
        if result["error"] is None
    ]

    st.write(
        f"**Images classified successfully:** "
        f"{len(successful_results)} / {len(results)}"
    )

    # Display each result separately
    for index, result in enumerate(results, start=1):

        st.divider()

        st.markdown(f"### Image {index}: {result['filename']}")

        image_column, result_column = st.columns(
            [1, 2],
            gap="large",
        )

        with image_column:
            if result["image"] is not None:
                st.image(
                    result["image"],
                    caption=result["filename"],
                    use_container_width=True,
                )
            else:
                st.error("Unable to display this image.")

        with result_column:

            if result["error"] is not None:
                st.error(
                    f"Could not classify this image: "
                    f"{result['error']}"
                )
                continue

            st.success(
                f"Predicted Class: "
                f"{result['predicted_class']}"
            )

            st.metric(
                "Prediction Confidence",
                f"{result['confidence']:.2f}%",
            )

            st.caption(
                "Confidence represents the model's estimated "
                "probability, not a guarantee of correctness."
            )

            with st.expander("🏆 View Top 3 Predictions"):

                for rank, prediction in enumerate(
                    result["top_predictions"],
                    start=1,
                ):

                    st.write(
                        f"**{rank}. {prediction['class']}** "
                        f"— {prediction['confidence']:.2f}%"
                    )

                    st.progress(
                        min(
                            max(
                                prediction["confidence"] / 100,
                                0.0,
                            ),
                            1.0,
                        )
                    )
