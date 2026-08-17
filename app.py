import streamlit as st
import pickle
import re
from PyPDF2 import PdfReader

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# --------------------------------------------------
# 2. Load Trained Models
# --------------------------------------------------

with open("tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("clf.pkl", "rb") as file:
    model = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

# --------------------------------------------------
# 3. Text Cleaning Function
# --------------------------------------------------

def clean_resume(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# --------------------------------------------------
# 4. PDF Text Extraction
# --------------------------------------------------

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

# --------------------------------------------------
# 5. ATS Score Function
# --------------------------------------------------

def calculate_ats_score(text):

    score = 0
    text_lower = text.lower()

    # Contact Information
    if re.search(r"\S+@\S+", text):
        score += 15

    if re.search(r"\+?\d[\d\s-]{8,}", text):
        score += 10

    # Standard Resume Sections
    sections = [
        "summary",
        "objective",
        "experience",
        "education",
        "skills",
        "projects",
        "certifications"
    ]

    for section in sections:

        if section in text_lower:
            score += 5

    # Technical Skills / Keywords
    skills = [
        "python",
        "sql",
        "java",
        "javascript",
        "html",
        "css",
        "machine learning",
        "data science",
        "deep learning",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "power bi",
        "excel",
        "git",
        "github",
        "docker",
        "aws",
        "linux"
    ]

    skill_count = 0

    for skill in skills:

        if skill in text_lower:
            skill_count += 1

    score += min(skill_count * 2, 20)

    # Resume Content Length
    word_count = len(text.split())

    if word_count >= 300:
        score += 10

    elif word_count >= 150:
        score += 5

    # Maximum Score = 100
    score = min(score, 100)

    return score

# --------------------------------------------------
# 6. Streamlit UI
# --------------------------------------------------

st.title("📄 AI Resume Screening System")

st.write(
    "Upload a resume and the Machine Learning model "
    "will predict the most suitable job category "
    "and calculate an ATS-style score."
)

st.divider()

# --------------------------------------------------
# 7. Resume Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

# --------------------------------------------------
# 8. Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Extract text from PDF
    resume_text = extract_text_from_pdf(
        uploaded_file
    )

    st.subheader("📋 Resume Preview")

    st.text_area(
        "Extracted Resume Text",
        resume_text,
        height=250
    )

    if st.button("🔍 Analyze Resume"):

        if not resume_text.strip():

            st.error(
                "Could not extract text from this PDF."
            )

        else:

            # ------------------------------------------
            # Clean Resume
            # ------------------------------------------

            cleaned_text = clean_resume(
                resume_text
            )

            # ------------------------------------------
            # TF-IDF Transformation
            # ------------------------------------------

            resume_vector = tfidf.transform(
                [cleaned_text]
            )

            # ------------------------------------------
            # ML Prediction
            # ------------------------------------------

            prediction = model.predict(
                resume_vector
            )

            # ------------------------------------------
            # Decode Category
            # ------------------------------------------

            category = encoder.inverse_transform(
                prediction
            )[0]

            # ------------------------------------------
            # Calculate ATS Score
            # ------------------------------------------

            ats_score = calculate_ats_score(
                resume_text
            )

            # ------------------------------------------
            # Display Job Category
            # ------------------------------------------

            st.success(
                f"🎯 Predicted Job Category: {category}"
            )

            # ------------------------------------------
            # Display ATS Score
            # ------------------------------------------

            st.subheader("📊 ATS Resume Score")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "ATS Score",
                    f"{ats_score}/100"
                )

            with col2:

                if ats_score >= 80:

                    rating = "Excellent"

                elif ats_score >= 60:

                    rating = "Good"

                elif ats_score >= 40:

                    rating = "Average"

                else:

                    rating = "Needs Improvement"

                st.metric(
                    "ATS Rating",
                    rating
                )

            # ------------------------------------------
            # Progress Bar
            # ------------------------------------------

            st.progress(
                ats_score / 100
            )

            # ------------------------------------------
            # ATS Message
            # ------------------------------------------

            if ats_score >= 80:

                st.success(
                    "🟢 Excellent ATS Compatibility"
                )

            elif ats_score >= 60:

                st.warning(
                    "🟡 Good ATS Compatibility - "
                    "Some improvements can be made."
                )

            elif ats_score >= 40:

                st.warning(
                    "🟠 Average ATS Compatibility - "
                    "Resume needs improvement."
                )

            else:

                st.error(
                    "🔴 Low ATS Compatibility - "
                    "Resume needs significant improvement."
                )

# --------------------------------------------------
# 9. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Built using Python, Scikit-learn, TF-IDF, "
    "Linear SVC, PyPDF2 and Streamlit"
)