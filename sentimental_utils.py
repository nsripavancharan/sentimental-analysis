from transformers import pipeline
import streamlit as st


@st.cache_resource
def load_model(model_name: str):
    """
    Load and cache the sentiment analysis model.
    """
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            return_all_scores=True
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        return None


def analyze_sentiment(text: str, classifier):
    """
    Analyze sentiment of the given text.
    """
    if not text or not text.strip():
        return None

    try:
        results = classifier(text)
        return results[0]  # return list of label-score dicts
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return None


def get_sentiment_emoji(label: str) -> str:
    """
    Map sentiment labels to emojis.
    """
    emoji_map = {
        "POSITIVE": "😊",
        "NEGATIVE": "😞",
        "NEUTRAL": "😐",
    }
    return emoji_map.get(label.upper(), "🤔")
