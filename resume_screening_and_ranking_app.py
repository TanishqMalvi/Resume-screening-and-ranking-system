import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import re
from io import BytesIO

# Function to set background
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: linear-gradient(145deg, rgba(9, 24, 48, 0.74), rgba(17, 43, 77, 0.67)),
                        url({image_url});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .main .block-container {{
            max-width: 1050px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        .hero-card {{
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(248, 250, 252, 0.95));
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(2, 8, 23, 0.18);
        }}
        .main-title {{
            margin: 0;
            font-size: 1.9rem;
            font-weight: 700;
            color: #0f172a;
            letter-spacing: -0.01em;
        }}
        .subtitle {{
            margin-top: 0.45rem;
            margin-bottom: 0;
            color: #334155;
            font-size: 0.95rem;
            font-weight: 400;
        }}
        .section-title {{
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 600;
            margin: 1rem 0 0.5rem 0;
            padding: 0.62rem 0.85rem;
            border-left: 3px solid #1d4ed8;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
        }}
        .stTextArea textarea {{
            background: #ffffff;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            color: #111827;
            line-height: 1.45;
        }}
        .stFileUploader {{
            background: rgba(255, 255, 255, 0.97);
            border-radius: 10px;
            padding: 0.72rem 0.85rem;
            border: 1px dashed #94a3b8;
        }}
        div[data-testid="stDataFrame"] {{
            background: rgba(255, 255, 255, 0.98);
            border-radius: 10px;
            padding: 0.55rem;
            border: 1px solid rgba(15, 23, 42, 0.1);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        }}
        .download-btn {{
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.62rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            color: #ffffff !important;
            text-decoration: none;
            background: #1e40af;
            border: 1px solid #1e3a8a;
            box-shadow: 0 6px 14px rgba(30, 64, 175, 0.3);
            transition: all 0.15s ease;
        }}
        .download-btn:hover {{
            background: #1d4ed8;
            transform: translateY(-1px) scale(1.005);
        }}
        .logo-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin: 0.45rem 0 1rem 0;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 10px;
            padding: 0.55rem 0.8rem;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.1);
        }}
        .logo-container img {{
            height: 54px;
            width: auto;
            object-fit: contain;
            filter: drop-shadow(0 3px 7px rgba(0, 0, 0, 0.14));
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.93), rgba(30, 41, 59, 0.95));
            border-right: 1px solid rgba(148, 163, 184, 0.24);
        }}
        [data-testid="stSidebar"] * {{
            color: #ffffff !important;
        }}
        </style>
        <div class="logo-container">
            <img src="https://i.ibb.co/DLfSPZy/aicte-logo.png">
            <img src="https://i.ibb.co/8Mbf7fy/edunet-logo.png">
            <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg">
            <img src="https://upload.wikimedia.org/wikipedia/commons/5/59/SAP_2011_logo.svg">
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

# Extract keywords from text
def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return set(words)

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words='english').fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Function to generate a download link
def generate_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a class="download-btn" href="data:file/csv;base64,{b64}" download="resume_ranking.csv">⬇ Download Ranked Resumes CSV</a>'
    return href

# Set background
set_background("https://images.pexels.com/photos/325229/pexels-photo-325229.jpeg?auto=compress&cs=tinysrgb&w=1600")

# Main Streamlit app
st.markdown(
    """
    <div class="hero-card">
        <h1 class="main-title">AI Resume Screening & Candidate Ranking System</h1>
        <p class="subtitle">A professional workflow to screen resumes and rank candidates against role-specific requirements.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("/home/runner/work/Resume-screening-and-ranking-system/Resume-screening-and-ranking-system/microsoft-logo.png", width=200)
st.sidebar.markdown("### Project Information")
st.sidebar.info("This AI-powered tool screens resumes based on job descriptions using NLP and TF-IDF.")

# Job Description Section
st.markdown('<div class="section-title">Job Description</div>', unsafe_allow_html=True)
job_description = st.text_area("Enter the job description")

# Upload Resume Section
st.markdown('<div class="section-title">Upload Resumes</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.markdown('<div class="section-title">Ranking Results</div>', unsafe_allow_html=True)
    resumes = []
    resume_keywords = {}
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
        resume_keywords[file.name] = extract_keywords(text)
    
    scores = rank_resumes(job_description, resumes)
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    st.write(results)
    st.markdown(generate_download_link(results), unsafe_allow_html=True)
    
    # Display extracted keywords
    st.markdown('<div class="section-title">Extracted Keywords from Resumes</div>', unsafe_allow_html=True)
    for name, keywords in resume_keywords.items():
        st.write(f"**{name}**: {', '.join(list(keywords)[:20])}")
