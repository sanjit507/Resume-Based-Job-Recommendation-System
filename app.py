import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import re
import spacy
from collections import Counter
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('generated_jobs.csv')

    # Combine relevant text fields into a single column
    df['combined_text'] = (
        df['description'].fillna('') + ' ' +
        df['requirements'].fillna('') + ' ' +
        df['responsibilities'].fillna('') + ' ' +
        df['required_skills'].fillna('')
    )

    # Clean the text
    df['cleaned_text'] = df['combined_text'].apply(lambda x: clean_text(x))
    return df


# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text


# Function to extract text from PDF files
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {e}")


# Function to extract text from DOCX files
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        raise ValueError(f"Error reading DOCX file: {e}")


# Generate TF-IDF vectors for all job descriptions
def generate_tfidf_vectors(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    return vectorizer, tfidf_matrix


# Function to get job recommendations based on resume text
def get_job_recommendations(resume_text, vectorizer, tfidf_matrix, df, top_n=5):
    # Vectorize the resume text
    resume_vector = vectorizer.transform([resume_text])

    # Compute cosine similarity between the resume and all job vectors
    similarities = cosine_similarity(resume_vector, tfidf_matrix).flatten()

    # Get indices of top-N most similar jobs
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Return the top-N recommended jobs
    return df.iloc[top_indices]


# Function to generate a word cloud
def generate_word_cloud(resume_text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(resume_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Word Cloud of Resume")
    st.pyplot(fig)




# Function to extract skills dynamically
def extract_skills(resume_text):
    # Process the text with spaCy
    doc = nlp(resume_text)

    # Extract noun chunks and named entities as potential skills
    skills = []
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ in ["NOUN", "PROPN"]:  # Focus on nouns and proper nouns
            skills.append(chunk.text.lower())

    # Add named entities (e.g., technologies, programming languages)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "TECH"]:  # Customize labels as needed
            skills.append(ent.text.lower())

    # Remove duplicates and empty strings
    skills = list(set(skills))
    skills = [skill for skill in skills if skill.strip()]

    return skills


# Function to generate an enhanced skills distribution graph
def generate_skills_graph(resume_text, top_n=7):
    # Dynamically extract skills from the resume
    skills = extract_skills(resume_text)

    if not skills:
        st.warning("No skills detected in the resume.")
        return

    # Count the frequency of each skill
    skills_count = Counter(skills)

    # Apply custom scoring: prioritize skills based on context and length
    contextual_scores = {}
    resume_text_lower = resume_text.lower()
    for skill in skills_count.keys():
        score = skills_count[skill]  # Start with raw frequency
        if f"proficient in {skill}" in resume_text_lower or f"expert in {skill}" in resume_text_lower:
            score += 5  # Bonus for skills explicitly mentioned as proficient/expert
        if len(skill.split()) > 1:  # Longer phrases (e.g., "machine learning") are weighted higher
            score += 3
        contextual_scores[skill] = score

    # Filter to show only the top N most relevant skills
    top_skills = sorted(contextual_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_skills_dict = {skill: score for skill, score in top_skills}

    # Normalize the scores to percentages
    total_score = sum(top_skills_dict.values())
    normalized_percentages = {skill: (score / total_score) * 100 for skill, score in top_skills_dict.items()}

    # Create an interactive bar chart with Plotly
    fig = px.bar(
        x=list(normalized_percentages.values()),
        y=list(normalized_percentages.keys()),
        orientation='h',
        title=f"Top {top_n} Skills Distribution",
        labels={'x': 'Percentage', 'y': 'Skill'},
        color=list(normalized_percentages.values()),  # Use color to represent percentage
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=50, b=50, l=150, r=50),
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)




# Streamlit app
def main():
    st.title("Resume-Based Job Recommendation System 🚀")
    st.write("Upload your resume to get personalized job recommendations and visualize your skills!")

    # Load the data
    df = load_data()

    # Generate TF-IDF vectors for job descriptions
    vectorizer, tfidf_matrix = generate_tfidf_vectors(df)

    # File uploader for resume
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX):", type=["pdf", "docx"])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        try:
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = extract_text_from_docx(uploaded_file)

            # Check if the extracted text is empty
            if not resume_text.strip():
                st.error("The uploaded file does not contain any readable text.")
                return

            # Preprocess the resume text
            resume_text = clean_text(resume_text)

            # Display extracted text

            generate_word_cloud(resume_text)
            # Generate graphs
            st.write("### Resume Visualization")
            generate_skills_graph(resume_text)

            # Get job recommendations
            recommendations = get_job_recommendations(resume_text, vectorizer, tfidf_matrix, df, top_n=10)

            # Display recommendations
            if not recommendations.empty:
                st.write("### Top 10 Recommended Jobs:")
                st.dataframe(recommendations[['title', 'company', 'location', 'category', 'salary']])
            else:
                st.warning("No matching jobs found.")
        except Exception as e:
            st.error(f"Error processing the file: {e}")


# Run the app
main()