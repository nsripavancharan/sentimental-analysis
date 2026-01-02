import streamlit as st
import pandas as pd

from sentiment_utils import (
    load_model,
    analyze_sentiment,
    get_sentiment_emoji
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


def display_single_result(results, model_name: str):
    """
    Display sentiment analysis results with metrics and table.
    """
    if not results:
        return

    st.subheader("Analysis Results")

    # Best sentiment (highest confidence)
    best_result = max(results, key=lambda x: x["score"])

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Sentiment",
            f"{get_sentiment_emoji(best_result['label'])} {best_result['label']}",
            f"{best_result['score']:.2%}"
        )

    with col2:
        st.metric(
            "Confidence Score",
            f"{best_result['score']:.2%}"
        )

    # Detailed table
    st.subheader("Detailed Scores")

    df = pd.DataFrame(results)
    df["Confidence"] = df["score"].apply(lambda x: f"{x:.2%}")
    df[""] = df["label"].apply(get_sentiment_emoji)
    df = df[["", "label", "Confidence"]]
    df.columns = ["", "Sentiment", "Confidence"]

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Model info
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Model:** {model_name}")
        st.write(
            "This RoBERTa-based model is fine-tuned on multiple datasets "
            "and provides state-of-the-art sentiment classification."
        )


def main():
    st.title("💬 Sentiment Analysis with Transformers")
    st.markdown(
        "Enter text below to analyze sentiment using a transformer-based model."
    )

    model_name = "siebert/sentiment-roberta-large-english"

    with st.spinner("Loading model... This may take a moment on first run."):
        classifier = load_model(model_name)

    if classifier is None:
        st.error("Failed to load the sentiment model. Please refresh.")
        return

    text_input = st.text_area(
        "Enter your text:",
        placeholder="Example: I love this product! It works flawlessly.",
        height=150,
        max_chars=1000
    )

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return

        with st.spinner("Analyzing sentiment..."):
            results = analyze_sentiment(text_input, classifier)

        if results:
            display_single_result(results, model_name)


if __name__ == "__main__":
    main()
