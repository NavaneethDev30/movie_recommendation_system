import pickle
import pandas as pd
import numpy as np

def load_and_inspect_data():
    try:
        # Load the data files
        with open('model/movie_list.pkl', 'rb') as f:
            movies = pickle.load(f)
        
        with open('model/similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
        
        # Basic inspection
        print("\n=== Movies DataFrame ===")
        print(f"Shape: {movies.shape}")
        print("\nFirst 5 movies:")
        print(movies.head())
        
        print("\n=== Similarity Matrix ===")
        print(f"Shape: {similarity.shape}")
        print("\nSample of similarity matrix:")
        print(similarity[:5, :5])  # First 5x5 entries
        
        # Check for null values
        print("\n=== Data Quality Check ===")
        print("Null values in movies:", movies.isnull().sum())
        
        return True
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        return False

if __name__ == "__main__":
    if load_and_inspect_data():
        print("\nData loaded successfully!")
    else:
        print("\nFailed to load data. Check your files.")