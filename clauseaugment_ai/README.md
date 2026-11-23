# ⚖️ ClauseAugment AI – Contract Clause Text Augmenter

> **A Generative AI-powered LegalTech system for contract clause text augmentation and ML classification.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io)
[![Gemini API](https://img.shields.io/badge/Google%20Gemini-API-yellow.svg)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview

ClauseAugment AI solves the problem of **limited diversity in contract clause training datasets** by using Google Gemini's generative AI to produce multiple legally-faithful variations of existing contract clauses. These augmented samples improve the robustness of ML classifiers trained on legal text.

### Key Features

| Feature | Description |
|---|---|
| 🤖 **AI Augmentation** | Google Gemini API generates 5 types of clause variations |
| 📊 **ML Classification** | TF-IDF + LR / Naive Bayes / Random Forest |
| 📈 **Model Evaluation** | Before vs after augmentation comparison with real metrics |
| 📂 **Dataset Management** | View, filter, search, download datasets |
| 🎨 **Professional UI** | Navy blue LegalTech themed Streamlit dashboard |
| 🔵 **Demo Mode** | Works without an API key using pre-written examples |

---

## 🏗️ Project Structure

```
clauseaugment_ai/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variable template
├── README.md                           # This file
│
├── config/
│   └── settings.py                     # App configuration & constants
│
├── data/
│   └── sample_clauses.csv              # 50-clause sample dataset
│
├── modules/
│   ├── augmentation/
│   │   ├── gemini_augmenter.py         # Gemini API integration
│   │   └── augmentation_techniques.py  # 5 technique prompts + demo data
│   ├── classification/
│   │   └── classifier.py               # ML classifiers (LR, NB, RF)
│   ├── evaluation/
│   │   └── metrics.py                  # Accuracy, Precision, Recall, F1
│   ├── nlp/
│   │   └── preprocessor.py             # NLTK preprocessing + similarity
│   ├── dataset/
│   │   └── manager.py                  # CSV load/save/filter/download
│   └── visualization/
│       └── charts.py                   # Plotly charts
│
└── pages/
    ├── home.py                         # Dashboard page
    ├── augmentation.py                 # Clause Augmentation page
    ├── classification.py               # Classification page
    ├── dataset_explorer.py             # Dataset Explorer page
    ├── evaluation.py                   # Model Evaluation page
    └── about.py                        # About page
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure API Key (Optional)

```bash
# Copy the example env file
copy .env.example .env

# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=AIza...your_key_here
```

Get your free Gemini API key at: https://makersuite.google.com/app/apikey

> **Note**: The app works in **Demo Mode** without an API key. Demo variations are pre-written examples clearly labelled as such.

### Step 3: Run the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📱 Application Sections

### 1. 🏠 Home Dashboard
- Overview metrics (original clauses, augmented clauses, categories, techniques)
- Category distribution charts
- Original vs augmented comparison
- Augmentation technique breakdown

### 2. ✨ Clause Augmentation
- **Manual input**: Type or paste a contract clause
- **CSV upload**: Batch augment multiple clauses
- Select clause category and augmentation techniques
- Generate 1–10 variations per clause
- Side-by-side comparison with cosine similarity scores
- Download augmented results as CSV

### 3. 🤖 Clause Classification
- Train classifiers on original / augmented / combined datasets
- Three algorithms: Logistic Regression, Naive Bayes, Random Forest
- Classify any new clause with confidence score
- Probability chart for all categories

### 4. 📂 Dataset Explorer
- View and filter original, augmented, and combined datasets
- Full-text search
- Category distribution charts
- Download any dataset as CSV

### 5. 📈 Model Evaluation
- One-click before vs after comparison
- Accuracy, Precision, Recall, F1-Score metrics
- Grouped bar chart and radar chart
- Side-by-side confusion matrices
- Detailed classification reports

### 6. ℹ️ About Project
- Project overview and architecture
- Technology stack
- Getting started guide
- Legal disclaimer

---

## 🧪 Augmentation Techniques

| Technique | Description |
|---|---|
| **Legal Paraphrasing** | Rewrite using different legal expressions |
| **Sentence Restructuring** | Change sentence structure (active/passive, clause order) |
| **Vocabulary Variation** | Replace words with legally equivalent alternatives |
| **Formal Language Variation** | Alternative professional drafting styles |
| **Structural Variation** | Reorganise internal structure (sub-clauses, ordering) |

---

## 🏷️ Supported Clause Categories

- Confidentiality
- Termination
- Indemnification
- Payment Terms
- Liability
- Intellectual Property
- Governing Law
- Data Protection

---

## ⚠️ Important Notes

1. **Legal Review Required**: All AI-generated variations must be reviewed by a qualified legal professional before use.
2. **No Legal Advice**: This tool is for research and demonstration purposes only.
3. **API Rate Limits**: Gemini API free tier has rate limits. The app handles these gracefully with fallback to demo mode.
4. **Actual Metrics**: Model evaluation results are real — not fabricated. Results depend on dataset size and quality.

---

## 📊 Sample Dataset

The bundled `data/sample_clauses.csv` contains 50 contract clauses across 8 categories:
- 5 clauses per category (Confidentiality, Termination, Indemnification, Payment Terms, Liability, IP, Governing Law, Data Protection) + 2 bonus clauses

Each clause is a realistic legal text suitable for demonstrating the augmentation pipeline.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.35 |
| Backend | Python 3.10+ |
| Generative AI | Google Gemini API (`gemini-1.5-flash`) |
| Machine Learning | Scikit-learn (TF-IDF, LR, NB, RF) |
| NLP | NLTK (tokenisation, lemmatisation) |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly, Matplotlib |
| Configuration | python-dotenv |

---

## 📄 License

This project is for educational and portfolio purposes.

---

*Built with ❤️ using Python, Streamlit, and Google Gemini API.*
