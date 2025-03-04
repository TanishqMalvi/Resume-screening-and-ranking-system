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
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
        }}
        .logo-container {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .logo-container img {{
            height: 80px;
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
    href = f'<a href="data:file/csv;base64,{b64}" download="resume_ranking.csv">Download Ranked Resumes</a>'
    return href

# Set background
set_background("https://source.unsplash.com/1600x900/?technology,abstract")

# Main Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

st.sidebar.image("https://source.unsplash.com/400x300/?hacker", width=200)
st.sidebar.markdown("### Project Information")
st.sidebar.info("This AI-powered tool screens resumes based on job descriptions using NLP and TF-IDF.")

# Job Description Section
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# Upload Resume Section
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
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
