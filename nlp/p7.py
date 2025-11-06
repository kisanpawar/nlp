"""
flipkart_sentiment.py
End-to-end sentiment analysis pipeline for Flipkart reviews.

Usage examples:
1) Train model:
   python flipkart_sentiment.py --train --data_path path/to/reviews.csv

2) Train with sample data (no dataset):
   python flipkart_sentiment.py --train

3) Predict single text:
   python flipkart_sentiment.py --predict "This phone is awesome!"

Model saved to ./models/sentiment_pipeline.joblib
"""

import os
import argparse
import joblib
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """Basic text normalization: lowercase, remove non-alpha, tokenise, remove stopwords, lemmatize."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # remove non-alphanumeric (keep spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load data. Expected CSV columns: 'review' and 'sentiment' (sentiment labels: 'positive','negative','neutral' or numeric)
    If csv_path is None, return a small sample dataset.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # try to standardize column names
        if 'review' not in df.columns:
            # look for common names
            for c in ['Review_Text','review_text','reviewText','text','Review']:
                if c in df.columns:
                    df.rename(columns={c: 'review'}, inplace=True)
                    break
        if 'sentiment' not in df.columns and 'rating' in df.columns:
            # convert rating to sentiment: 4-5 positive, 3 neutral, 1-2 negative
            df['sentiment'] = df['rating'].apply(lambda r: 'positive' if r >= 4 else ('neutral' if r == 3 else 'negative'))
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("CSV must contain columns 'review' and 'sentiment' (or 'rating').")
        df = df[['review','sentiment']].dropna()
        return df
    else:
        # sample small dataset
        data = {
            'review': [
                "Very good phone, battery life is amazing and camera is excellent.",
                "Terrible. The product stopped working in 2 days. Waste of money.",
                "Average experience. The phone is okay for the price.",
                "Excellent quality! Highly recommended.",
                "Not satisfied. The screen had dead pixels and the speaker is low.",
                "Value for money. Works as expected.",
                "Bad packaging. Received a damaged box but product was fine.",
                "Love it! Fast and smooth. Good purchase."
            ],
            'sentiment': [
                'positive', 'negative', 'neutral', 'positive', 'negative', 'positive', 'negative', 'positive'
            ]
        }
        return pd.DataFrame(data)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['review_clean'] = df['review'].apply(preprocess_text)
    return df

def build_pipeline() -> Pipeline:
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(solver='liblinear', max_iter=1000))
    ])
    return pipeline

def train_and_evaluate(df: pd.DataFrame, model_path: str = './models/sentiment_pipeline.joblib'):
    df = prepare_features(df)
    X = df['review_clean']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_pipeline()

    # Optional small grid search (fast)
    param_grid = {
        'tfidf__max_features': [2000, 5000],
        'clf__C': [0.5, 1.0]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=0)
    print("Training model (this may take a minute)...")
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    y_pred = best.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best, model_path)
    print(f"Model saved to {model_path}")
    return best

def load_model(model_path: str = './models/sentiment_pipeline.joblib'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
    return joblib.load(model_path)

def predict_text(texts, model=None, model_path: str = './models/sentiment_pipeline.joblib'):
    if model is None:
        model = load_model(model_path)
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [preprocess_text(t) for t in texts]
    preds = model.predict(cleaned)
    probs = model.predict_proba(cleaned) if hasattr(model, 'predict_proba') else None
    results = []
    for i, t in enumerate(texts):
        res = {'text': t, 'cleaned': cleaned[i], 'predicted_sentiment': preds[i]}
        if probs is not None:
            res['probabilities'] = dict(zip(model.classes_, probs[i].tolist()))
        results.append(res)
    return results

def main(args):
    if args.train:
        df = load_data(args.data_path)
        train_and_evaluate(df, args.model_path)
    elif args.predict:
        results = predict_text(args.predict, model=None, model_path=args.model_path)
        for r in results:
            print("Text:", r['text'])
            print("Cleaned:", r['cleaned'])
            print("Predicted:", r['predicted_sentiment'])
            if 'probabilities' in r:
                print("Prob:", r['probabilities'])
            print("-"*40)
    else:
        print("No action specified. Use --train or --predict. See help (--help).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help="Train model")
    parser.add_argument('--data_path', type=str, default=None, help="Path to reviews CSV")
    parser.add_argument('--model_path', type=str, default='./models/sentiment_pipeline.joblib', help="Path to save/load model")
    parser.add_argument('--predict', type=str, default=None, help="Predict sentiment for a text")
    args = parser.parse_args()
    main(args)
