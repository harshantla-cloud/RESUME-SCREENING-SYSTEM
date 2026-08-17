# 📄 AI Resume Screening System

An end-to-end Machine Learning web app that reads a resume (PDF), predicts the most suitable **job category**, and generates an **ATS (Applicant Tracking System) compatibility score** — all in real time using a Streamlit interface.

![Predicted Job Category & ATS Score](./Project%20Images/Predict%20with%20ATS%20Score.jpeg)

---

## 🚀 Live Demo Preview

| Upload Resume | Extracted Preview |
|---|---|
| ![App Home](./Project%20Images/App%20Image.jpeg) | ![Resume Preview](./Project%20Images/Load%20Resume%20Inside%20model.jpeg) |

---

## 📌 Overview

Recruiters and hiring platforms rely heavily on ATS systems to filter resumes before a human ever sees them. This project simulates that pipeline end-to-end:

1. A user uploads their resume as a **PDF**.
2. Text is extracted and cleaned.
3. A trained **ML classification model** predicts the most relevant **job category** (e.g., Data Science, Java Developer, HR, DevOps Engineer, etc.).
4. A custom **ATS scoring engine** evaluates the resume on contact info, standard sections, keyword/skill density, and content length, then returns a score out of 100 with a compatibility rating.

The goal was to build a complete, deployable data science project — from raw dataset to a working, interactive product — rather than just a notebook.

---

## ✨ Features

- 📤 **PDF Resume Upload** — drag-and-drop or browse, up to 200MB
- 🧹 **Automated Text Cleaning** — removes URLs, emails, and noise before inference
- 🎯 **Job Category Prediction** — classifies resumes into 25 job categories using TF-IDF + Linear SVC
- 📊 **ATS Resume Score (0–100)** — checks for contact details, key resume sections, technical keywords, and word count
- 🟢🟡🔴 **ATS Rating** — Excellent / Good / Average / Needs Improvement, with a visual progress bar
- 👀 **Live Resume Text Preview** before analysis

---

## 🧠 How It Works

```
Resume (PDF)
     │
     ▼
Text Extraction (PyPDF2)
     │
     ▼
Text Cleaning (regex-based preprocessing)
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Linear SVC Classifier ──► Predicted Job Category
     │
     ▼
Rule-based ATS Scoring Engine ──► ATS Score + Rating
```

---

## 🏗️ Model Training (Summary)

The classification model was trained in `analysis.ipynb` on the [`UpdatedResumeDataSet.csv`](./UpdatedResumeDataSet.csv), which contains ~960 labeled resumes across **25 job categories** (Java Developer, Testing, DevOps Engineer, Python Developer, Data Science, HR, Mechanical Engineer, and more).

Key steps:
- Balanced class distribution via oversampling to handle category imbalance
- Text cleaning (URLs, mentions, hashtags, special characters removed)
- Label encoding of job categories
- TF-IDF vectorization of resume text
- Compared multiple classifiers (KNN, Random Forest, Logistic Regression) — **Linear SVC (One-vs-Rest)** gave the best performance and was selected as the final model
- Final artifacts exported with `pickle`: `tfidf.pkl`, `clf.pkl`, `encoder.pkl`

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML / Data | Scikit-learn, Pandas, NumPy |
| NLP | TF-IDF Vectorizer |
| Model | Linear SVC (One-vs-Rest) |
| PDF Parsing | PyPDF2 |
| Web App | Streamlit |
| Visualization (EDA) | Matplotlib, Seaborn |

---

## 📂 Project Structure

```
Resume Screening App/
├── app.py                       # Streamlit application
├── analysis.ipynb               # EDA + model training notebook
├── UpdatedResumeDataSet.csv     # Training dataset
├── tfidf.pkl                    # Saved TF-IDF vectorizer
├── clf.pkl                      # Saved trained classifier
├── encoder.pkl                  # Saved label encoder
└── Project Images/              # Screenshots used in this README
```

---

## ⚙️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/resume-screening-app.git
   cd resume-screening-app
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit scikit-learn pandas numpy PyPDF2 matplotlib seaborn
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open the local URL Streamlit gives you, upload a resume PDF, and click **Analyze Resume**.

---

## 📊 ATS Scoring Logic

The ATS score (out of 100) is calculated based on:

| Criteria | Points |
|---|---|
| Contact info (email, phone) | up to 25 |
| Standard resume sections (Summary, Experience, Education, Skills, Projects, Certifications) | up to 35 |
| Technical keywords/skills detected (Python, SQL, ML, Git, AWS, etc.) | up to 20 |
| Resume length / content depth | up to 10 |

**Rating scale:** 80+ Excellent · 60–79 Good · 40–59 Average · Below 40 Needs Improvement

---

## 🔮 Future Improvements

- [ ] Support DOCX resume uploads
- [ ] Deep learning-based classifier (BERT embeddings) for higher accuracy
- [ ] Job description matching — compare resume against a specific JD
- [ ] Resume improvement suggestions (missing sections/keywords)
- [ ] Deploy on Streamlit Cloud / HuggingFace Spaces with public link

---

## 🙋 About

Built by **Harsh** as part of a Machine Learning portfolio project, showcasing an end-to-end pipeline — from data preprocessing and model training to a deployed, interactive ML application.

📫 Connect on [LinkedIn](#) | 💻 [GitHub](https://github.com/harshantla-cloud)

---

*Built using Python, Scikit-learn, TF-IDF, Linear SVC, PyPDF2 and Streamlit.*
