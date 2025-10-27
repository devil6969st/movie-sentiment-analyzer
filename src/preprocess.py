import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import joblib

df = pd.read_csv(r"C:\Users\shank\nlp_project\data\IMDB Dataset.csv")
df['label'] = df['sentiment'].map({'positive':1,'negative':0})

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text

df['cleaned_review'] = df["review"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'],
                                                    df['label'], test_size = 0.2, random_state=42, stratify=df['label'])

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1500)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

joblib.dump(model, "model_logreg.joblib")
joblib.dump(model, "vectorizer.joblib")
