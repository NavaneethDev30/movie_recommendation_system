import pickle
import pandas as pd

# Load the model
movies = pickle.load(open('model/movies.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

def recommend(movie_title):
    if movie_title not in movies['title'].values:
        print("❌ Movie not found.")
        return []

    index = movies[movies['title'] == movie_title].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recommended_titles = []
    for i in sorted_movies:
        recommended_titles.append(movies.iloc[i[0]]['title'])
    
    return recommended_titles

# Example usage
movie_name = input("Enter a movie name: ")
recommendations = recommend(movie_name)
if recommendations:
    print("\n🎬 Recommended Movies:")
    for title in recommendations:
        print(" -", title)
