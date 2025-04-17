import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'resume_id': [1, 2, 3],
    'resume_text': [
        "Experienced data scientist with skills in python, machine learning",
        "Software developer with expertise in java, cloud computing, and project machine learning",
        "Data analyst with proficiency in sql, python"
    ]
}

job_description = "Looking for a data scientist skilled in python, machine learning"
df = pd.DataFrame(data)
print("Resumes:\n", df)

documents = df['resume_text'].tolist()
documents.append(job_description)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
df['similarity_scores'] = similarity_scores

print("\nResume similarity scores:\n", df[['resume_id', 'similarity_scores']])

threshold = 0.2
matching_resumes = df[df['similarity_scores'] >= threshold]
print("\nResumes matching the job requirements:\n", matching_resumes[['resume_id', 'similarity_scores']])
