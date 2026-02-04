import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

try:
    from xgboost import XGBClassifier
    use_xgb = True
except ImportError:
    use_xgb = False
    print("xgboost not available — using sklearn GradientBoostingClassifier")

df = pd.read_csv("final_data/labeled_political_sentiment.csv")
X = df["text"].astype(str)
y = df["sentiment_label"].astype(str).str.lower().str.strip()

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Label mapping:", dict(enumerate(le.classes_)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

if use_xgb:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_tfidf, y_train)
else:
    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_train_tfidf.toarray(), y_train)

y_pred = model.predict(X_test_tfidf if use_xgb else X_test_tfidf.toarray())
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))