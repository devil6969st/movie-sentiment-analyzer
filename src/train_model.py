from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'],
                                                    df['label'], test_size = 0.2, random_state=42, stratify=df['label'])

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1500)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)





