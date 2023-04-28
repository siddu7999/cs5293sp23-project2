import unittest
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from project2.cusine import data_load, process_data, vectorize, training, closest_cusine


class TestProject2(unittest.TestCase):
    
    def setUp(self):
        self.data = [{"cuisine": "italian", "ingredients": ["garlic", "tomatoes", "olive oil", "basil"]},
                     {"cuisine": "mexican", "ingredients": ["avocado", "lime", "cilantro", "onion"]},
                     {"cuisine": "chinese", "ingredients": ["soy sauce", "ginger", "garlic", "green onions"]}]
        with open('test_data.json', 'w') as f:
            json.dump(self.data, f)
        self.X = np.array(['garlic, onion, tomato', 'turmeric, cumin, coriander', 'ginger, chili, coconut'])
        self.vectorizer, self.X_vectorized = vectorize(self.X)
        self.model = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine')
        self.model.fit(self.X_vectorized)
        
    def test_data_load(self):
        loaded_data = data_load('test_data.json')
        self.assertEqual(loaded_data, self.data)

    def test_process_data(self):
        X_expected = ['garlic, tomatoes, olive oil, basil', 'avocado, lime, cilantro, onion', 'soy sauce, ginger, garlic, green onions']
        y_expected = ['italian', 'mexican', 'chinese']
        X, y = process_data(self.data)
        self.assertEqual(X, X_expected)
        self.assertEqual(y, y_expected)

    def test_training(self):
        model = training(self.X_vectorized, k=2, algorithm='brute', metric='cosine')
        self.assertIsInstance(model, NearestNeighbors)
        self.assertEqual(model.n_neighbors, 2)
        self.assertEqual(model.algorithm, 'brute')
        self.assertEqual(model.metric, 'cosine')

    def test_closest_cusine(self):
        ingredients = ['onion', 'tomato', 'ginger']
        closest_cuisines = closest_cusine(self.model, self.vectorizer, ingredients, k=2)
        self.assertIsInstance(closest_cuisines, list)
        self.assertEqual(len(closest_cuisines), 2)
        self.assertIsInstance(closest_cuisines[0], dict)
        self.assertSetEqual(set(closest_cuisines[0].keys()), {'id', 'score'})
        self.assertIsInstance(closest_cuisines[0]['id'], str)
        self.assertTrue(closest_cuisines[0]['id'].isdigit())
        self.assertIsInstance(closest_cuisines[0]['score'], str)
        self.assertEqual(len(closest_cuisines[0]['score']), 4)


if __name__ == '__main__':
    unittest.main()
