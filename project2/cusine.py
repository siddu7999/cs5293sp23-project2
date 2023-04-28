import argparse
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

def data_load(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def process_data(data):
    X = [', '.join(row['ingredients']).lower() for row in data]
    y = [row['cuisine'] for row in data]
    return X, y

def vectorize(X):
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    return vectorizer, X_vectorized

def training(X_vectorized, k=5, algorithm='brute', metric='cosine'):
    model = NearestNeighbors(n_neighbors=k, algorithm=algorithm, metric=metric)
    model.fit(X_vectorized)
    return model

def closest_cusine(model, vectorizer, ingredients, k=5):
    ingredients = [ingredient.lower() for ingredient in ingredients]
    ingredients_str = ', '.join(ingredients)
    ingredients_vectorized = vectorizer.transform([ingredients_str])
    distances, indices = model.kneighbors(ingredients_vectorized)
    closest_cuisines = []
    for i in range(k):
        cuisine = {
            "id": str(indices[0][i]),
            "score": f"{distances[0][i]:.2f}"
        }
        closest_cuisines.append(cuisine)
    return closest_cuisines
