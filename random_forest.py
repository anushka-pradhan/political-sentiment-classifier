import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("final_data/labeled_political_sentiment.csv")
print(df["sentiment_label"].value_counts(normalize=True))

texts = df["text"].astype(str).tolist()
labels = df["sentiment_label"].tolist()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_embeddings(text_list, batch_size=16):
    all_embeddings = []
    model.eval()
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(batch_embeddings)
        print(f"Processed batch {i // batch_size + 1} of {len(text_list) // batch_size + 1}", end="\r")
    return np.vstack(all_embeddings)

print("Generating DistilBERT embeddings...")
X = get_embeddings(texts)
y = labels

print("\nEmbeddings generated!")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("Evaluating...")
y_pred = rf.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
