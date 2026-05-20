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
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {{
            font-family: 'Poppins', sans-serif;
        }}

        .stApp {{
            background: linear-gradient(135deg, rgba(5, 8, 25, 0.85), rgba(10, 31, 68, 0.78)),
                        url({image_url});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .main .block-container {{
            max-width: 1080px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        .hero-card {{
            background: rgba(255, 255, 255, 0.09);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1.2rem;
            backdrop-filter: blur(8px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }}
        .main-title {{
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
        }}
        .subtitle {{
            margin-top: 0.45rem;
            margin-bottom: 0;
            color: #dbeafe;
            font-size: 0.98rem;
            font-weight: 400;
        }}
        .section-title {{
            color: #ffffff;
            font-size: 1.2rem;
            font-weight: 600;
            margin: 1rem 0 0.5rem 0;
            padding: 0.55rem 0.85rem;
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(6px);
        }}
        .stTextArea textarea {{
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #cbd5e1;
            border-radius: 10px;
        }}
        .stFileUploader {{
            background: rgba(255, 255, 255, 0.92);
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
            border: 1px dashed #93c5fd;
        }}
        div[data-testid="stDataFrame"] {{
            background: rgba(255, 255, 255, 0.96);
            border-radius: 12px;
            padding: 0.5rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
        }}
        .download-btn {{
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.6rem 1rem;
            border-radius: 10px;
            font-weight: 600;
            color: #ffffff !important;
            text-decoration: none;
            background: linear-gradient(90deg, #2563eb, #0ea5e9);
            box-shadow: 0 6px 16px rgba(14, 116, 144, 0.3);
            transition: transform 0.15s ease;
        }}
        .download-btn:hover {{
            transform: translateY(-1px);
        }}
        .logo-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 22px;
            margin: 0.6rem 0 1.1rem 0;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 14px;
            padding: 0.6rem 0.9rem;
            backdrop-filter: blur(7px);
        }}
        .logo-container img {{
            height: 62px;
            width: auto;
            object-fit: contain;
            filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2));
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(10, 31, 68, 0.88), rgba(15, 23, 42, 0.92));
            border-right: 1px solid rgba(255, 255, 255, 0.14);
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
set_background("https://source.unsplash.com/1600x900/?technology,abstract")

# Main Streamlit app
st.markdown(
    """
    <div class="hero-card">
        <h1 class="main-title">AI Resume Screening & Candidate Ranking System</h1>
        <p class="subtitle">Screen resumes faster with cleaner ranking insights and a polished project interface.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("https://source.unsplash.com/400x300/?hacker", width=200)
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
    st.subheader("Extracted Keywords from Resumes")
    for name, keywords in resume_keywords.items():
        st.write(f"**{name}**: {', '.join(list(keywords)[:20])}")
