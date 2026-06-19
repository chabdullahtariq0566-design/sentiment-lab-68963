# Legal Notice Multi-Class Document Classifier
### Programming For AI -- End-Term Lab Assessment | Spring 2026
**Riphah International University, Lahore Campus**

---

## Project Description
A multi-class text classifier that categorises short legal notices (50-200 words) into three categories:
- **A:** Contract Dispute
- **B:** Intellectual Property Claim
- **C:** Regulatory Compliance

Two classical ML models (Logistic Regression and Multinomial Naive Bayes) are compared across two feature representations (Bag-of-Words and TF-IDF), with all experiments tracked via MLflow.

---

## Project Structure
```
sentiment-lab-[studentID]/
├── data/
│   ├── raw/               # Original legal_notices.csv
│   └── processed/         # Cleaned CSV after preprocessing
├── notebooks/
│   └── sentiment_analysis.ipynb   # Main notebook (run this)
├── src/
│   ├── preprocess.py      # Text cleaning pipeline
│   ├── features.py        # BoW and TF-IDF vectorisers
│   └── evaluate.py        # Metrics, confusion matrices, timing
├── config.json            # All hyperparameters (no hardcoding in code)
├── requirements.txt       # Pinned dependencies
├── mlruns/                # MLflow experiment tracking artefacts
├── results/               # Confusion matrix images, charts, CSV
└── README.md              # This file
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/sentiment-lab-<studentID>.git
cd sentiment-lab-<studentID>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (first time only)
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## How to Run

### Run the Jupyter Notebook
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```
Run all cells top to bottom. The notebook handles:
- EDA and visualisations (saved to `results/`)
- Preprocessing pipeline
- Model training and evaluation
- MLflow experiment logging

### View MLflow UI (after running the notebook)
```bash
mlflow ui
```
Open `http://127.0.0.1:5000` in your browser to see all logged runs.

---

## Configuration File (`config.json`)
All hyperparameters are stored in `config.json` and read programmatically. No values are hardcoded in the source code.

| Key | Value | Description |
|-----|-------|-------------|
| `random_seed` | 42 | Global random seed for reproducibility |
| `test_size` | 0.2 | 20% of data used for testing |
| `max_features` | 5000 | Vocabulary size for vectorisers |
| `model_1.C` | 1.0 | Regularisation parameter for Logistic Regression |
| `model_2.alpha` | 1.0 | Laplace smoothing for Naive Bayes |

---

## Results Summary

| Configuration | Accuracy | F1 (Macro) | F1 (Weighted) |
|---|---|---|---|
| LogisticRegression + BoW | 1 | 1 | 1 |
| LogisticRegression + TF-IDF | 1 | 1 | 1 |
| NaiveBayes + BoW | 1 | 1 | 1 |
| NaiveBayes + TF-IDF | 1 | 1 | 1 |

*(Exact values printed in Cell 10 of the notebook after execution)*

**Best configuration:** Logistic Regression + TF-IDF

---

## AI Usage Statement
AI tools (Claude by Anthropic) were used for code structure suggestions and docstring formatting. All design decisions, problem framing, and analysis were completed independently. See Task 5.3 in the notebook for full AI Usage Statement.
