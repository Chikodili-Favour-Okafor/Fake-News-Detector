"""
Fake News Detection Dashboard
-----------------------------
A Streamlit interface for a trained TF-IDF + Linear SVM pipeline.

Run locally with:
    python -m streamlit run app.py
"""

# ============================================================
# 1. IMPORT LIBRARIES
# Import tools for file paths, timestamps, data handling,
# loading the saved model, and building the Streamlit interface.
# ============================================================

from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 2. LOCATE THE SAVED MODEL
# Find the directory containing app.py and build the path
# to the saved pipeline without relying on the terminal location.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fake_news_pipeline.joblib"


# ============================================================
# 3. CONFIGURE THE PAGE
# Set the browser tab information and use a wide layout
# to accommodate the dashboard sidebar and main content.
# ============================================================

st.set_page_config(
    page_title="Fake News Detection Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 4. APPLY CUSTOM STYLING
# Add subtle colors, spacing, and card-like containers
# to create a more polished analytics dashboard.
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
        }

        .dashboard-subtitle {
            color: #64748b;
            font-size: 1rem;
            margin-top: -0.5rem;
            margin-bottom: 1.5rem;
        }

        .section-heading {
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.07);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 1rem;
            border-radius: 0.75rem;
        }

        div.stButton > button {
            border-radius: 0.55rem;
            min-height: 2.7rem;
        }

        div[data-testid="stTextArea"] textarea {
            line-height: 1.6;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. INITIALIZE SESSION STATE
# Preserve input fields, the latest result, and prediction
# history across Streamlit reruns during the current session.
# ============================================================

if "headline" not in st.session_state:
    st.session_state.headline = ""

if "article_text" not in st.session_state:
    st.session_state.article_text = ""

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# ============================================================
# 6. DEFINE THE SAMPLE ARTICLE
# Provide illustrative text so users can explore the app
# without having to find an article before their first test.
# This example does not have a guaranteed prediction.
# ============================================================

SAMPLE_HEADLINE = (
    "City Council Announces a New Public Transport Initiative"
)

SAMPLE_ARTICLE = """
The city council has announced plans for a new public transport
initiative intended to improve access to major residential areas.

According to the illustrative announcement, the proposal will
be reviewed at a public meeting before a final decision is made.
Residents will be invited to share feedback on the proposed routes,
service schedules, and implementation timeline.

Further details would be published after the review process.
"""


# ============================================================
# 7. DEFINE INPUT BUTTON ACTIONS
# The sample button fills both input fields. The reset button
# clears the fields and the latest result, but keeps history.
# ============================================================

def load_sample():
    st.session_state.headline = SAMPLE_HEADLINE
    st.session_state.article_text = SAMPLE_ARTICLE
    st.session_state.latest_result = None


def reset_inputs():
    st.session_state.headline = ""
    st.session_state.article_text = ""
    st.session_state.latest_result = None


def clear_history():
    st.session_state.prediction_history = []


# ============================================================
# 8. LOAD THE TRAINED PIPELINE
# Cache the model so Streamlit can reuse it during reruns.
# The pipeline already includes text preprocessing, TF-IDF,
# and the trained Linear SVM classifier.
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "The saved model was not found. Make sure "
        "'fake_news_pipeline.joblib' is in the project folder."
    )
    st.stop()

except Exception:
    st.error(
        "The model could not be loaded. Check that "
        "'text_preprocessing.py' is available and that the "
        "installed package versions are compatible."
    )
    st.stop()


# ============================================================
# 9. BUILD THE SIDEBAR
# Present project information, the model's recorded test
# metrics, and an important limitation of the predictions.
# Update these metrics if your final evaluation changes.
# ============================================================

with st.sidebar:
    st.title("📰 Fake News Detector")

    st.caption("Machine-learning text classification dashboard")

    st.divider()

    st.subheader("About the model")

    st.write(
        "This application uses TF-IDF text features and a "
        "Linear Support Vector Machine (Linear SVM) to "
        "classify submitted articles."
    )

    st.write("**Text representation:** TF-IDF")
    st.write("**Classifier:** Linear SVM")
    st.write("**Prediction labels:** FAKE / REAL")

    st.divider()

    st.subheader("Recorded test performance")

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric("Accuracy", "93%")
    metric_col2.metric("FAKE F1-score", "94%")

    metric_col1.metric("FAKE Precision", "92%")
    metric_col2.metric("FAKE Recall", "95%")

    st.caption(
        "These are the results from the previously evaluated "
        "test set, not a new evaluation performed by this app."
    )

    st.divider()

    st.subheader("How it works")

    st.write(
        "1. Enter a headline and/or article text.\n\n"
        "2. Submit the article for analysis.\n\n"
        "3. The saved pipeline processes the text and predicts "
        "a label.\n\n"
        "4. Review the result and, if needed, export your "
        "session's prediction history."
    )

    st.warning(
        "This tool is for educational purposes. A prediction "
        "is not independent verification of an article's truth."
    )


# ============================================================
# 10. DISPLAY THE MAIN DASHBOARD HEADER
# Introduce the application and explain what users can do.
# ============================================================

st.title("📰 Fake News Detection")

st.markdown(
    """
    <div class="dashboard-subtitle">
        Analyze an article using a trained machine-learning model.
        Enter its headline, article text, or both to get a prediction.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 11. DISPLAY TOP-LEVEL DASHBOARD METRICS
# Summarize the recorded model accuracy and the number of
# predictions made in the current session.
# ============================================================

history_count = len(st.session_state.prediction_history)

total_col, accuracy_col, model_col = st.columns(3)

total_col.metric(
    "Articles Analyzed",
    history_count,
    help="Number of successful predictions in this session."
)

accuracy_col.metric(
    "Recorded Test Accuracy",
    "93%",
    help="Accuracy recorded during the previous model evaluation."
)

model_col.metric(
    "Model",
    "Linear SVM",
    help="The classifier used by the saved pipeline."
)

st.divider()


# ============================================================
# 12. CREATE THE ARTICLE INPUT SECTION
# Provide fields for the headline and article body, plus
# buttons to load example text or reset the current inputs.
# ============================================================

st.subheader("Analyze an article")

st.write(
    "Enter the information below. You can use the sample "
    "article to test the interface."
)

button_col1, button_col2, button_col3 = st.columns(
    [1, 1, 4]
)

with button_col1:
    st.button(
        "Load sample",
        on_click=load_sample,
        use_container_width=True
    )

with button_col2:
    st.button(
        "Reset inputs",
        on_click=reset_inputs,
        use_container_width=True
    )

title = st.text_input(
    "Article headline",
    key="headline",
    placeholder="Enter the headline here..."
)

article_text = st.text_area(
    "Article text",
    key="article_text",
    height=260,
    placeholder="Paste the full article text here..."
)

character_count = len(article_text)

st.caption(
    f"Article body: {character_count:,} characters"
)


# ============================================================
# 13. VALIDATE INPUT AND GENERATE A PREDICTION
# When the user clicks the button, combine the headline and
# body, reject empty submissions, and call the saved pipeline.
# ============================================================

if st.button(
    "Analyze article",
    type="primary",
    use_container_width=True
):

    combined_text = f"{title} {article_text}".strip()

    if not combined_text:
        st.warning(
            "Please enter a headline, article text, or both."
        )

    else:
        try:
            with st.spinner("Analyzing the article..."):

                prediction = model.predict([combined_text])[0]

            # Store the successful prediction and its details.
            timestamp = datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )

            result = {
                "timestamp": timestamp,
                "headline": title.strip() or "(No headline entered)",
                "prediction": str(prediction),
                "article_characters": character_count
            }

            st.session_state.latest_result = result

            st.session_state.prediction_history.insert(
                0, result.copy()
            )

        except Exception:
            st.error(
                "The prediction could not be completed. "
                "Check the model and preprocessing setup."
            )


# ============================================================
# 14. DISPLAY THE LATEST PREDICTION
# Show the latest result in a clear section after a prediction
# has been made. The result remains visible across reruns until
# another prediction is made or the inputs are reset.
# ============================================================

latest_result = st.session_state.latest_result

if latest_result is not None:

    st.divider()
    st.subheader("Prediction result")

    result_col1, result_col2 = st.columns([2, 1])

    with result_col1:

        if latest_result["prediction"] == "FAKE":
            st.error(
                "**Model classification: FAKE**\n\n"
                "The model classified this submitted text as "
                "belonging to the FAKE class."
            )

        elif latest_result["prediction"] == "REAL":
            st.success(
                "**Model classification: REAL**\n\n"
                "The model classified this submitted text as "
                "belonging to the REAL class."
            )

        else:
            st.info(
                f"Model classification: "
                f"{latest_result['prediction']}"
            )

    with result_col2:
        st.metric(
            "Article body length",
            f"{latest_result['article_characters']:,} chars"
        )

        st.caption(
            f"Analyzed at: {latest_result['timestamp']}"
        )

    st.info(
        "This is a machine-learning prediction based on text "
        "patterns learned during training. It does not establish "
        "whether the article is factually true or false."
    )


# ============================================================
# 15. DISPLAY PREDICTION HISTORY
# Show successful predictions from the current session.
# Users can review previous results and clear the history.
# ============================================================

st.divider()
st.subheader("Recent prediction history")

history = st.session_state.prediction_history

if history:

    history_df = pd.DataFrame(history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    download_col, clear_col = st.columns([2, 1])

    with download_col:
        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="Download history as CSV",
            data=csv_data,
            file_name="fake_news_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    with clear_col:
        st.button(
            "Clear history",
            on_click=clear_history,
            use_container_width=True
        )

else:
    st.info(
        "No predictions yet. Submit an article above to "
        "start building your session history."
    )


# ============================================================
# 16. DISPLAY THE FOOTER
# Finish with a brief project note and a reminder about
# responsible interpretation of the model's predictions.
# ============================================================

st.divider()

st.caption(
    "Fake News Detection | TF-IDF + Linear SVM | "
    "Educational machine-learning project"
)