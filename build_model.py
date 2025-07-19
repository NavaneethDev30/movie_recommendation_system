import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
# Merge on 'title'
movies = movies.merge(credits, on='title')

# Select useful columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Handle missing values
movies.dropna(inplace=True)

# Convert stringified lists into real lists
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Top 3 cast members
def convert_cast(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

movies['cast'] = movies['cast'].apply(convert_cast)

# Get the director's name
def get_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['crew'] = movies['crew'].apply(get_director)

# Convert overview from string to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Merge all lists into a single "tags" column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Vectorize
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save files
pickle.dump(new_df, open("model/movie_list.pkl", "wb"))
pickle.dump(similarity, open("model/similarity.pkl", "wb"))

print("✅ Model files created: model/movie_list.pkl & model/similarity.pkl")
