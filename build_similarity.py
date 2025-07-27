import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load movie metadata CSV (adjust filename/path if needed)
movies = pd.read_csv('tmdb_5000_movies.csv')

# Example: combine text features (genres, overview) for similarity
movies['combined_features'] = movies['genres'] + ' ' + movies['overview']

# Vectorize the combined features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'].fillna(''))

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Save the similarity matrix as a pickle file
with open('model/similarity.pkl', 'wb') as f:
    pickle.dump(similarity_matrix, f)

# Optional: save the movie list for your app
with open('model/movie_list.pkl', 'wb') as f:
    pickle.dump(movies, f)

print("Similarity matrix and movie list saved successfully.")

