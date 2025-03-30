Project Description
The Resume-Based Job Recommendation System is a web application designed to help users upload their resumes and receive personalized job recommendations. It also provides visualizations such as word count, skills distribution (bar chart), and a word cloud of the resume text. The system uses advanced natural language processing (NLP) techniques to extract skills and match them with job descriptions from a dataset.

https://github.com/sanjit507/Resume-Based-Job-Recommendation-System/blob/e149b997c5e1d2e0f29826f92570b6646ec5f135/Screenshot%202025-03-30%20145457.png


Technologies Used

Backend Framework :
Streamlit : A lightweight Python library for building interactive web applications quickly.
Data Processing :
Pandas : For loading and manipulating the job dataset.

Scikit-learn : For generating TF-IDF vectors and calculating cosine similarity between resumes and job descriptions.

Natural Language Processing (NLP) :
spaCy : For extracting skills and named entities from the resume text.
Regex : For cleaning and preprocessing text data.

Visualizations :
Plotly : For creating interactive bar charts of skill distributions.
WordCloud : For generating word clouds to visualize frequently used words in the resume.

File Handling :
PyPDF2 : For extracting text from PDF files.
python-docx : For extracting text from DOCX files.

