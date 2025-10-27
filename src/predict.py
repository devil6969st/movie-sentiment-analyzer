import joblib
import re
from bs4 import BeautifulSoup

model = joblib.load(r'C:\Users\shank\nlp_project\model\tfidf_logreg.joblib')
vectorizer = joblib.load(r'C:\Users\shank\nlp_project\model\tfidf_vectorizer.joblib')

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text

def predict_sentiment(review):
    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return 1 if pred == 1 else 0

print(predict_sentiment("this is a good movie"))