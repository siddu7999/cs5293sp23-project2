import cusine
import json
import argparse
from cusine import data_load, process_data,vectorize,training,closest_cusine
def main():
    parser = argparse.ArgumentParser(description='Find the closest cuisines based on a list of ingredients.')
    parser.add_argument('--file', type=str, default='yummly.json', help='Path to the JSON file containing the recipe data.')
    parser.add_argument('--N', type=int, default=5, help='Number of closest cuisines to display.')
    parser.add_argument('--ingredient', type=str, nargs='+', help='List of ingredients to use for finding the closest cuisines.')
    args = parser.parse_args()

    data = data_load(args.file)
    X, y = process_data(data)
    vectorizer, X_vectorized = vectorize(X)
    model = training(X_vectorized, k=args.N)

    closest_cuisines = closest_cusine(model, vectorizer, args.ingredient, k=args.N)

    output = {
        "cuisine": y[int(closest_cuisines[0]['id'])],
        "score": closest_cuisines[0]['score'],
        "closest": closest_cuisines
    }

    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
