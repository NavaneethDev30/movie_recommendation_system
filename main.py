import pickle
import pandas as pd
import requests
import streamlit as st
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Load models
movies = pickle.load(open('model/movie_list.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

def fetch_poster(movie_title):
    """Fetch poster using movie title"""
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = requests.get(url)
        data = response.json()
        poster_path = data['results'][0]['poster_path']
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return ""

def recommend(movie):
    if movie not in movies['title'].values:
        return [], []

    idx = movies[movies['title'] == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        title = movies.iloc[i[0]]['title']
        poster = fetch_poster(title)
        recommended_movies.append(title)
        recommended_posters.append(poster)

    return recommended_movies, recommended_posters

# Streamlit UI
st.title('Movie Recommender System')

selected_movie = st.selectbox("Choose a movie:", movies['title'].values)

if st.button('Recommend'):
    try:
        names, posters = recommend(selected_movie)
        if names:
            col1, col2, col3, col4, col5 = st.columns(5)
            columns = [col1, col2, col3, col4, col5]
            for col, name, poster in zip(columns, names, posters):
                with col:
                    st.text(name)
                    st.image(poster)
        else:
            st.error("No recommendations found. Try another movie.")
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
