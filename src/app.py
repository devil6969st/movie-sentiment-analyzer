import streamlit as st
import joblib
import re
from bs4 import BeautifulSoup
from predict import predict_sentiment

try:
    model = joblib.load(r"C:\Users\shank\nlp_project\model\tfidf_logreg.joblib")
    vectorizer = joblib.load(r"C:\Users\shank\nlp_project\model\tfidf_vectorizer.joblib")
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found. Make sure the paths are correct.")
    st.stop()

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text

st.title("🎬 Movie Review Sentiment Analyzer")
review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        prediction = predict_sentiment(review)
        if prediction == 1:
            st.success("Prediction: Positive 😊")
        else:
            st.error("Prediction: Negative 😞")
