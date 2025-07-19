import os
import pickle
import requests
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

API_KEY = os.getenv('TMDB_API_KEY')
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/500x750?text=No+Poster"
MAX_RETRIES = 3
TIMEOUT = 15  # seconds

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.header("🎬 Movie Recommendation Engine")

@st.cache_data
def load_data():
    try:
        with open('model/movie_list.pkl', 'rb') as f:
            movies = pickle.load(f)
        with open('model/similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
        return movies, np.array(similarity)
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

movies, similarity = load_data()

def fetch_poster(movie_id):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}",
                params={"api_key": API_KEY},
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                poster_path = response.json().get('poster_path')
                if poster_path:
                    return f"{POSTER_BASE_URL}{poster_path}"
                else:
                    return PLACEHOLDER_IMAGE
        except requests.exceptions.RequestException:
            if attempt == MAX_RETRIES - 1:
                return PLACEHOLDER_IMAGE
            time.sleep(0.5)
    return PLACEHOLDER_IMAGE

def recommend(movie_title, top_n=5):
    try:
        movie_index = movies[movies['title'] == movie_title].index[0]
        similar_indices = np.argsort(-similarity[movie_index])[1:top_n*5]  # get extra

        recommendations = []
        movie_candidates = [movies.iloc[idx] for idx in similar_indices]

        # Fetch posters in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_poster, movie.movie_id): movie for movie in movie_candidates}
            for future in as_completed(futures):
                movie = futures[future]
                poster = future.result()
                if poster:
                    recommendations.append({
                        'title': movie.title,
                        'poster': poster,
                        'id': movie.movie_id
                    })
                    if len(recommendations) >= top_n:
                        break
        return recommendations if recommendations else None
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return None

selected_movie = st.selectbox(
    "Select a movie:",
    movies['title'].values,
    index=0
)

if st.button("Get Recommendations"):
    with st.spinner("Finding recommendations..."):
        start_time = time.time()
        recommendations = recommend(selected_movie)
        st.write(f"Found in {time.time() - start_time:.2f} seconds")

    if recommendations:
        cols = st.columns(5)
        for col, movie in zip(cols, recommendations[:5]):
            with col:
                st.image(movie['poster'], use_container_width=True)
                st.markdown(f"**{movie['title']}**")
    else:
        st.error("No recommendations found. Try another movie or check your connection.")
        if st.button("Retry"):
            st.experimental_rerun()

st.markdown("---")
st.caption("Data provided by The Movie Database")
