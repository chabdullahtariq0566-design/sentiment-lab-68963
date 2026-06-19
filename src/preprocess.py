"""
preprocess.py
=============
Reusable text preprocessing pipeline for the Legal Notice Classifier.
Applies NLP cleaning steps in a fixed, justified order.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources if not already present
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)


def preprocess_text(text: str) -> str:
    """
    Clean and normalize a single legal notice string.

    Applies the following steps in order:
        1. HTML tag removal
        2. Lowercasing
        3. Punctuation & special character removal
        4. Tokenisation
        5. Stopword removal (NLTK English stopwords)
        6. Lemmatisation (chosen over stemming — see justification below)

    Lemmatisation Justification:
        Legal text contains domain-specific vocabulary (e.g., "licenses",
        "licensing", "licensed"). Stemming would reduce all three to "licens",
        losing readability and sometimes producing non-words. Lemmatisation
        produces real base forms ("license"), which is preferable for
        interpretability during viva and for TF-IDF weight analysis.

    Stopword Choice:
        NLTK's default English stopword list is used because it covers common
        function words ("the", "and", "is") that carry no discriminative signal
        for multi-class legal document classification.

    Args:
        text (str): Raw notice string.

    Returns:
        str: Cleaned, space-joined token string ready for vectorisation.
    """

    # --- Step 1: Remove HTML tags ---
    # Legal notices may contain HTML artifacts if scraped from web portals.
    text = re.sub(r"<[^>]+>", " ", text)

    # --- Step 2: Lowercase ---
    # Ensures "Contract" and "contract" are treated identically by the vectoriser.
    text = text.lower()

    # --- Step 3: Remove punctuation and special characters ---
    # Punctuation (e.g., commas, dollar signs, parentheses) adds noise for
    # bag-of-words models and does not carry class-discriminative information.
    text = re.sub(r"[^a-z\s]", " ", text)

    # --- Step 4: Tokenise ---
    # Split text into individual word tokens for per-token processing.
    tokens = text.split()

    # --- Step 5: Remove stopwords ---
    # NLTK English stopwords filter out high-frequency, low-information words.
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    # --- Step 6: Lemmatise ---
    # Reduce tokens to their dictionary base form (e.g., "infringes" → "infringe").
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_series(series):
    """
    Apply preprocess_text to an entire Pandas Series.

    Args:
        series (pd.Series): Series of raw text strings.

    Returns:
        pd.Series: Series of cleaned text strings.
    """
    return series.fillna("").apply(preprocess_text)
