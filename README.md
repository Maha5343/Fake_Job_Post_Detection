# Fake_Job_Post_Detection
Machine learning project to detect fake job postings using NLP techniques and classification models.
# 🎯 Fake Job Post Detection

A final year B.Tech project designed to detect fraudulent job postings using machine learning and natural language processing (NLP). With the rise of online job platforms, job scams are increasing, and this system aims to assist job seekers by flagging suspicious job listings based on data-driven predictions.

---

## 🚧 Problem Statement

Online job portals like LinkedIn, Indeed, and others are flooded with thousands of job listings every day. Unfortunately, many of them are **fraudulent or scam posts** created to mislead applicants, steal data, or gain unauthorized access to sensitive information.

Manual identification of these fake posts is inefficient, and victims often fall prey to appealing but suspicious job offers.

---

## 🎯 Project Objective

To develop an intelligent system that can automatically analyze job postings and classify them as **real** or **fake** using machine learning models trained on labeled data.

---

## 📁 Dataset Overview

- **Source**: Kaggle ([Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction))
- **Total Records**: ~18,000
- **Target Variable**: `fraudulent` (0 = Real, 1 = Fake)
- **Key Features**:
  - `title`
  - `location`
  - `department`
  - `company_profile`
  - `description`
  - `requirements`
  - `benefits`
  - `industry`, `function`, etc.

---

## 🧠 Methodology

### 1. Data Preprocessing
- Handling missing and inconsistent data
- Text cleaning (removing HTML tags, special characters)
- Removing stopwords and stemming
- Encoding categorical variables

### 2. Feature Extraction
- **TF-IDF Vectorization**: Converts textual features into numerical format
- Extracted from fields like `description`, `requirements`, `company_profile`

### 3. Model Training
- Trained multiple classifiers:
  - Logistic Regression
  - Random Forest
  - Multinomial Naive Bayes
  - Support Vector Machine (SVM)
- **Cross-validation** for robust performance checking

### 4. Model Evaluation
- Confusion Matrix
- Accuracy, Precision, Recall, F1-score
- ROC Curve & AUC Score

---

## 📈 Key Results

| Model               | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 94.8%    | 95.3%     | 93.7%  | 94.5%    |
| Random Forest      | 95.2%    | 96.0%     | 94.1%  | 95.0%    |
| Naive Bayes        | 91.3%    | 92.5%     | 90.2%  | 91.3%    |

---

## 📊 Visualizations

- Word clouds for fake vs real job descriptions
- Correlation heatmaps
- Count plots for class distributions
- ROC Curves for model comparison

---

## 📂 Project Structure

Fake_Job_Post_Detection/
│
├── data/ # Dataset files (CSV)
├── notebooks/ # EDA and model building notebooks
├── models/ # Trained model files (joblib/pickle)
├── visuals/ # Output images and graphs
├── app/ # Flask/Streamlit app files (optional)
├── requirements.txt # Python package list
├── README.md
└── .gitignore


---

## 🧩 Challenges Faced

- Class imbalance: Fake job posts are significantly fewer than real ones
- Unstructured text: Required advanced text preprocessing
- Misleading fake jobs that mimic real structure

---

## 🚀 Future Improvements

- Use **BERT / Transformer-based models** for deep semantic understanding
- Build a **web application** where users can paste job posts and get predictions
- Integrate with browser extensions to auto-detect suspicious posts

---

## 💡 Applications

- Career portals to pre-filter job listings
- Browser plugins for scam detection
- Background checks by HR and staffing agencies


---

## 📜 License

This project is intended for **educational and research purposes** only.

