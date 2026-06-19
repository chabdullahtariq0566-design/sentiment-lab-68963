"""
features.py
===========
Feature extraction module for the Legal Notice Classifier.
Implements Bag-of-Words (BoW) and TF-IDF representations.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_bow_vectorizer(max_features: int = 5000):
    """
    Build a Bag-of-Words CountVectorizer.

    BoW represents each document as a word frequency count vector.
    Simple and fast, but ignores term importance across documents.

    Args:
        max_features (int): Maximum vocabulary size. Defaults to 5000.

    Returns:
        CountVectorizer: Unfitted sklearn vectoriser instance.
    """
    return CountVectorizer(max_features=max_features)


def build_tfidf_vectorizer(max_features: int = 5000):
    """
    Build a TF-IDF TfidfVectorizer with sublinear term frequency scaling.

    TF-IDF down-weights terms that appear in many documents (low discriminative
    power) and up-weights rare but informative terms. sublinear_tf=True applies
    log(1 + tf) scaling, preventing very frequent terms from dominating.

    Args:
        max_features (int): Maximum vocabulary size. Defaults to 5000.

    Returns:
        TfidfVectorizer: Unfitted sklearn vectoriser instance.
    """
    return TfidfVectorizer(max_features=max_features, sublinear_tf=True)


def fit_transform_vectorizer(vectorizer, X_train, X_test):
    """
    Fit vectoriser on training data and transform both train and test sets.

    Fitting only on training data prevents data leakage — test vocabulary
    must not influence the feature space learned during training.

    Args:
        vectorizer: A CountVectorizer or TfidfVectorizer instance.
        X_train (pd.Series or list): Training text documents.
        X_test (pd.Series or list): Test text documents.

    Returns:
        tuple: (X_train_vec, X_test_vec, fitted_vectorizer)
    """
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def print_top_terms(vectorizer, n: int = 20, representation_name: str = ""):
    """
    Print the top N terms by overall weight/frequency from a fitted vectoriser.

    For TF-IDF: uses sum of IDF-weighted scores across vocabulary.
    For BoW: prints top N terms alphabetically (vocabulary is unordered by default).
    This gives a quick sanity check that domain-relevant terms dominate.

    Args:
        vectorizer: A fitted CountVectorizer or TfidfVectorizer.
        n (int): Number of top terms to display. Defaults to 20.
        representation_name (str): Label for display purposes.
    """
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(vectorizer, "idf_"):
        # TF-IDF: rank by IDF score (higher = rarer = more informative)
        scores = vectorizer.idf_
        top_indices = np.argsort(scores)[::-1][:n]
        print(f"\n--- Top {n} terms by IDF weight ({representation_name}) ---")
        for idx in top_indices:
            print(f"  {feature_names[idx]:<30} IDF={scores[idx]:.4f}")
    else:
        # BoW: no inherent ranking; show first N from vocabulary
        print(f"\n--- First {n} vocabulary terms ({representation_name}) ---")
        for term in feature_names[:n]:
            print(f"  {term}")
