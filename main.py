import pandas as pd
import os

# Load dataset
file_path = os.path.join(
    os.path.dirname(__file__),
    "dataset",
    "fake_news_dataset.csv"
)

df = pd.read_csv(file_path)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print("Number of rows and columns:", df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())
# ============================================
# STEP 4: DATA PREPROCESSING & TEXT CLEANING
# ============================================

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

# Fill missing values
df["source"] = df["source"].fillna("Unknown")
df["author"] = df["author"].fillna("Unknown")

# Combine title and text
df["content"] = df["title"] + " " + df["text"]

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["content"] = df["content"].apply(clean_text)

print("\n========== PREPROCESSING COMPLETE ==========")
print(df[["content", "label"]].head())
# ============================================
# STEP 6: FEATURE EXTRACTION
# ============================================

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["content"])
y = df["label"]

print("\n========== FEATURE EXTRACTION COMPLETE ==========")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)
# ============================================
# STEP 7: TRAIN-TEST SPLIT
# ============================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
# ============================================
# STEP 8: MODEL TRAINING
# ============================================

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# MODEL 1: LOGISTIC REGRESSION
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# MODEL 2: NAIVE BAYES
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

print("\n========== MODEL TRAINING COMPLETE ==========")
print("Logistic Regression: Trained")
print("Naive Bayes: Trained")
# ============================================
# STEP 9: MODEL EVALUATION
# ============================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Logistic Regression Prediction
logistic_pred = logistic_model.predict(X_test)

print("\n========== LOGISTIC REGRESSION ==========")
print("Accuracy:", accuracy_score(y_test, logistic_pred))
print("Precision:", precision_score(y_test, logistic_pred, average="weighted"))
print("Recall:", recall_score(y_test, logistic_pred, average="weighted"))
print("F1-Score:", f1_score(y_test, logistic_pred, average="weighted"))
print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))


# Naive Bayes Prediction
naive_bayes_pred = naive_bayes_model.predict(X_test)

print("\n========== NAIVE BAYES ==========")
print("Accuracy:", accuracy_score(y_test, naive_bayes_pred))
print("Precision:", precision_score(y_test, naive_bayes_pred, average="weighted"))
print("Recall:", recall_score(y_test, naive_bayes_pred, average="weighted"))
print("F1-Score:", f1_score(y_test, naive_bayes_pred, average="weighted"))
print("Confusion Matrix:")
print(confusion_matrix(y_test, naive_bayes_pred))
# ============================================
# STEP 10: MODEL COMPARISON
# ============================================

logistic_accuracy = accuracy_score(y_test, logistic_pred)
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_pred)

print("\n========== MODEL COMPARISON ==========")
print("Logistic Regression:", round(logistic_accuracy, 4))
print("Naive Bayes:", round(naive_bayes_accuracy, 4))

if naive_bayes_accuracy > logistic_accuracy:
    best_model = naive_bayes_model
    best_model_name = "Naive Bayes"
    best_accuracy = naive_bayes_accuracy
else:
    best_model = logistic_model
    best_model_name = "Logistic Regression"
    best_accuracy = logistic_accuracy

print("\n========== BEST MODEL ==========")
print("Best Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy, 4))
# ============================================
# STEP 11: SAVE BEST MODEL
# ============================================

import joblib

joblib.dump(best_model, "best_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n========== BEST MODEL SAVED ==========")
print("Best model saved successfully.")
print("TF-IDF vectorizer saved successfully.")