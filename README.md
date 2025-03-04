# AI Resume Screening & Candidate Ranking System

This is an AI-powered Resume Screening & Candidate Ranking System that analyzes resumes against a job description using Natural Language Processing (NLP) and TF-IDF (Term Frequency-Inverse Document Frequency). It calculates the similarity between resumes and the job description to rank candidates accordingly.

## Features
- Extracts text from uploaded PDF resumes
- Uses TF-IDF vectorization to process text data
- Computes cosine similarity to rank resumes based on relevance to the job description
- Displays extracted keywords from resumes
- Provides an option to download ranked resumes as a CSV file
- Interactive web-based UI using Streamlit

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/ai-resume-screening.git
   cd ai-resume-screening
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Dependencies
Ensure you have the following Python libraries installed:
```sh
streamlit
PyPDF2
pandas
scikit-learn
base64
re
```

## How to Run

Run the Streamlit application with the following command:
```sh
streamlit run app.py
```

## Usage
1. Enter the job description in the text area.
2. Upload multiple resumes in PDF format.
3. The system will extract text from resumes and rank them based on relevance.
4. View the ranking results and extracted keywords.
5. Download the ranked resumes as a CSV file.

## Project Information
This project is developed as part of the AICTE Internship on AI: Transformative Learning with TechSaksham, a joint CSR initiative of Microsoft & SAP.

## License
This project is licensed under the MIT License.

## Contributing
Feel free to fork this repository, make improvements, and submit pull requests.

## Contact
For any issues or contributions, reach out via GitHub issues.



 
