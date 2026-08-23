# Fake News Detection System

## Project Overview

The Fake News Detection System is a machine learning project that classifies news articles as fake or real.

The project uses Natural Language Processing (NLP) techniques to clean and process news text, extract important features using TF-IDF, and train machine learning models for classification.

## Dataset

The dataset contains news articles with information such as:

- Title
- Text
- Source
- Author
- Label

The `label` column is used as the target variable for classification.

The dataset is stored inside the `dataset` folder.

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- Joblib
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes

## Project Workflow

1. Load the dataset
2. Explore the dataset
3. Check missing values and duplicate records
4. Clean and preprocess the text
5. Remove stopwords
6. Apply stemming using Porter Stemmer
7. Convert text into numerical features using TF-IDF
8. Split the dataset into training and testing data
9. Train Logistic Regression and Naive Bayes models
10. Evaluate both models
11. Compare model performance
12. Save the best-performing model
13. Save the TF-IDF vectorizer

## Text Preprocessing

The text data is processed using:

- Lowercase conversion
- Removal of non-alphabetic characters
- Stopword removal
- Porter stemming
- Combining the news title and article text

## Machine Learning Models

### Logistic Regression

Logistic Regression is used as one of the classification models for detecting whether a news article is fake or real.

### Multinomial Naive Bayes

Multinomial Naive Bayes is another classification model used with TF-IDF features.

The model with the better accuracy is selected as the best model.

## Saved Files

The project saves two important files:

- `best_model.pkl` – trained best-performing machine learning model
- `tfidf_vectorizer.pkl` – trained TF-IDF vectorizer

These files can be used later for making predictions without retraining the model.

## Project Structure

```text
Fake_News_Detection/
│
├── dataset/
│   └── fake_news_dataset.csv
│
├── best_model.pkl
├── tfidf_vectorizer.pkl
├── main.py
├── README.md
└── requirements.txt