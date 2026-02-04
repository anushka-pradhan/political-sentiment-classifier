import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    use_xgb = True
except ImportError:
    use_xgb = False
    print("xgboost not available — using sklearn GradientBoostingClassifier")


train_df = pd.read_csv("final_data/labeled_political_sentiment.csv")

X_full = train_df["text"].astype(str)
y_full = train_df["sentiment_label"].astype(str).str.lower().str.strip()


X_train_text, X_val_text, y_train_raw, y_val_raw = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    random_state=42,
    stratify=y_full
)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_val = le.transform(y_val_raw)

print("Label mapping:", dict(enumerate(le.classes_)))

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf = vectorizer.transform(X_val_text)


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
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train_tfidf.toarray(), y_train)


y_train_pred = model.predict(X_train_tfidf if use_xgb else X_train_tfidf.toarray())
train_acc = accuracy_score(y_train, y_train_pred)
print("TRAINING PERFORMANCE")
print("Training Accuracy:", train_acc)

y_val_pred = model.predict(X_val_tfidf if use_xgb else X_val_tfidf.toarray())


print("\nVALIDATION PERFORMANCE")
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

print("\nClassification Report:\n", classification_report(y_val, y_val_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

test_df = pd.read_csv("final_data/hand_labeled_data.csv")

X_test_text = test_df["text"].astype(str)
y_test_raw = test_df["sentiment_label"].astype(str).str.lower().str.strip()
y_test = le.transform(y_test_raw)

X_test_tfidf = vectorizer.transform(X_test_text)

y_test_pred = model.predict(X_test_tfidf if use_xgb else X_test_tfidf.toarray())

print("\nFINAL TEST PERFORMANCE")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

print("\nTrain class distribution:\n", y_train_raw.value_counts(normalize=True))
print("\nValidation class distribution:\n", y_val_raw.value_counts(normalize=True))
print("\nTest class distribution:\n", y_test_raw.value_counts(normalize=True))
