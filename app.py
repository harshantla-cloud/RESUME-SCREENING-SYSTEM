import streamlit as st
import pickle, re, base64
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

BASE = Path(__file__).parent

# ================= BACKGROUND =================

bg = BASE / "background.png"

if bg.exists():
    img = base64.b64encode(bg.read_bytes()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image:
        linear-gradient(
            rgba(10,15,25,0.82),
            rgba(10,15,25,0.82)
        ),
        url("data:image/png;base64,{img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

# ================= MODELS =================

with open(BASE / "models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open(BASE / "models/clf.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE / "models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ================= FUNCTIONS =================

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf(file):
    reader = PdfReader(file)
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )


def extract_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


def ats_score(text):
    t = text.lower()
    score = 0

    if re.search(r"\S+@\S+", text):
        score += 15

    if re.search(r"\+?\d[\d\s-]{8,}", text):
        score += 10

    sections = [
        "summary", "objective", "experience",
        "education", "skills", "projects",
        "certifications"
    ]

    score += sum(5 for x in sections if x in t)

    skills = [
        "python", "sql", "java", "javascript",
        "html", "css", "machine learning",
        "data science", "deep learning",
        "pandas", "numpy", "scikit-learn",
        "tensorflow", "power bi", "excel",
        "git", "github", "docker", "aws", "linux"
    ]

    score += min(sum(x in t for x in skills) * 2, 20)

    words = len(text.split())

    if words >= 300:
        score += 10
    elif words >= 150:
        score += 5

    return min(score, 100)

# ================= UI =================

st.title("📄 AI Resume Screening System")

st.write(
    "Upload your resume and optionally compare it "
    "with a Job Description."
)

st.divider()

resume_file = st.file_uploader(
    "📤 Upload Resume",
    type=["pdf", "docx"]
)

job_description = st.text_area(
    "💼 Job Description (Optional)",
    height=160,
    placeholder="Paste ONLY the Job Description here..."
)

# ================= ANALYSIS =================

if resume_file:

    if resume_file.name.lower().endswith(".pdf"):
        resume = extract_pdf(resume_file)
    else:
        resume = extract_docx(resume_file)

    if not resume.strip():
        st.error("❌ Could not extract text from resume.")
        st.stop()

    with st.expander("📋 Resume Preview"):
        st.text_area(
            "Extracted Resume",
            resume,
            height=180
        )

    if st.button("🔍 Analyze Resume", type="primary"):

        text = clean(resume)

        # -------- Job Prediction --------

        vector = tfidf.transform([text])
        prediction = model.predict(vector)

        category = encoder.inverse_transform(
            prediction
        )[0]

        # -------- ATS --------

        score = ats_score(resume)

        st.divider()

        st.subheader("🎯 Predicted Job Category")
        st.info(f"💼 **{category}**")
        st.caption("✓ ML prediction completed successfully")

        # -------- ATS Result --------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "📊 ATS-Style Score",
                f"{score}/100"
            )

        with col2:

            rating = (
                "Excellent" if score >= 80 else
                "Good" if score >= 60 else
                "Average" if score >= 40 else
                "Needs Improvement"
            )

            st.metric(
                "ATS Rating",
                rating
            )

        st.progress(score / 100)

        # -------- JD Matching --------

        if job_description.strip():

            st.divider()
            st.subheader("💼 Resume–Job Match")

            jd_text = clean(job_description)

            resume_vec = tfidf.transform([text])
            jd_vec = tfidf.transform([jd_text])

            match = cosine_similarity(
                resume_vec,
                jd_vec
            )[0][0] * 100

            match = round(match, 2)

            st.metric(
                "🎯 Job Match Score",
                f"{match}%"
            )

            st.progress(
                min(match / 100, 1.0)
            )

            if match >= 75:
                st.success("🟢 Strong Match")
            elif match >= 50:
                st.warning("🟡 Moderate Match")
            else:
                st.error("🔴 Low Match")

        else:

            st.info(
                "💡 Add a Job Description to calculate "
                "Resume–Job Match."
            )

# ================= FOOTER =================

st.divider()

st.caption(
    "Built with Python • Scikit-learn • TF-IDF • "
    "Linear SVC • PyPDF2 • python-docx • Streamlit"
)