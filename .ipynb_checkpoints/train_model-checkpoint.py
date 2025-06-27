import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("disease_treatment_500.csv")
df["Disease"] = df["Disease"].str.lower().str.strip()
df["Treatment"] = df["Treatment"].str.strip()

# Input and Output
X = df["Disease"]
y = df["Treatment"]

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF + Random Forest
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42, oob_score=True))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, "disease_treatment_model.pkl")

print("✅ Model trained and saved as disease_treatment_model.pkl")
