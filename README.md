# CS5293SP23_project-2
## SIDDARTHA SODUM
### OU_I'D:113581755

## Project description:  
The objective of this project is to develop a cuisine predictor that groups foods according to their ingredients and forecasts the cuisine type of a new item in order to assist a hotel chain in increasing menu revenues. The data sets, which are made available by Yummly.com, comprise a comprehensive list of all potential dishes, a list of each dish's ingredients, a unique identity, and the cuisine in which it is served. The project calls for developing an application that analyzes a user's list of components to determine the type of cuisine it will be and then locates the top-N nearby foods. The procedure for creating the project is training or indexing the food data, asking the user to enter ingredients, asking the user to enter ingredients,
and returning the IDs of the meals that are the closest to the user's input.
## Required Packages:
   ### Json: 
   The json package is a built-in Python module that provides methods for working with JSON (JavaScript Object Notation) data.
   ### scikit-learn (Installation : pip install scikit-learn)
   The scikit-learn package provides the CountVectorizer and NearestNeighbors classes that are used in the program.
   ### Numpy:
   NumPy (short for "Numerical Python") is a powerful Python package for scientific computing and data analysis.
   ### Argparse (Installation : pip install argparse)
   The argparse package provides an easy way to parse command line arguments in the program.
   ### Pytest:
   pytest is a Python testing framework that provides a number of features to make writing and running tests easier and more powerful.
## Install and RUN:

mkdir my_project

cd my_project

are used to create a directory 

git clone "link of your repository" is used to download the project 

To manage the virtual environment “pipenv shell” is used.

Install the required packages (scikit-learn, numpy, unittest)

“pipenv run python project2.py –N 5 --ingredient paprika
                                                           --ingredient banana
                                                           --ingredient “rice krispies”

The above command is used to run the project2.py

“pipenv run python -m pytest -v”

is used to run the test functions.
      
## Functions used in the project:
The code has two code parts, in which one is cusine.py and the other is project2.py 
Predictor.py/
### data_load:
The load_data function opens the provided file in read mode ('r') using the open() function, then reads and parses the contents of the file into a Python object using the json.load() function. The function returns the final Python object after reading and parsing the file's contents. In conclusion, the load_data function in Python offers a practical method for loading data from a JSON file.
### process_data:
Preprocess_data receives a list of dictionaries, each of which has the keys "ingredients" and "cuisine" and represents a recipe. The function then preprocesses the data by constructing a string from all of the recipe's ingredients using a comma as a separator and changing its case to lowercase. X and y are two independent lists that contain the preprocessed data.
### vectorize:
The function creates a CountVectorizer object and initializes it before assigning it to the vectorizer variable. The CountVectorizer object's fit_transform() method is then used on the X data to fit the vectorizer to the data and return the transformed data in sparse matrix format. The variable X_vectorized receives the transformed data as its value.
The trained vectorizer object and the modified X_vectorized data are both returned by the function, which can then be utilized for additional processing or as input to a machine learning model for prediction or training.
### training:
The function first creates a scikit-learn nearest neighbors model object and assigns it to the model variable. Based on the supplied inputs, the n_neighbors, method, and metric hyperparameters are set for the model object. Then the input data X_vectorized is passed to the fit() method of the nearest neighbors model object, which trains the model on the input data to identify the closest neighbors based on the supplied hyperparameters.
The function then returns the trained model object, which may be used to analyse further data or make predictions about brand-new pieces of information.

I have compared the accuracies of several machine learning models like the naive bayes, knn classifier, svc classifier for the given yummly.json data set, in which the knn turned out to be more accurate and efficient comparitively.
### closest_cusine:
This function changes all of the input ingredients to lower case, unites them into a string separated by commas, and then vectorizes the string using the transform() method of the vectorizer object, returning a sparse matrix of the vectorized input data.
The vectorized input data is then passed to the kneighbors() method of the nearest neighbors model object, which provides the indices and distances of the input data's k nearest neighbors in the trained model.
Finally, the function generates a list of dictionaries, each containing an id for the cuisine and a score derived from the model's distances. The list is then sorted by the score in ascending order and the first k elements are returned as the k closest cuisines.
### Project2.py/:
The argparse module, which enables command-line arguments such as the path to the JSON file containing the recipe data, the number of closest cuisines to display, and a list of ingredients to use for finding the closest cuisines, is first initialized by the main() function.
The user-specified JSON file is next loaded with the recipe data using the load_data() function from the culinary module. Next, the data is preprocessed by changing the ingredient lists to lowercase and separating them with commas using the preprocess_data() function.
The preprocessed data is then converted into a sparse matrix of features using the CountVectorizer class from scikit-learn by calling the vectorize_data() function.
Following that, the user-specified number of nearest neighbors is trained on the vectorized data using the train_model() method.
The trained model and vectorizer are used to determine the closest cuisines based on the input ingredients in the last step, after which the output is displayed in JSON format.

“parser.add_argument('--N', type=int, default=5, help='Number of closest cuisines to display.')” 

This line defines a command-line argument named --N that expects an integer value and sets a default value of 5 if no value is provided, and provides a brief help message for the argument.

“parser.add_argument('--ingredient', type=str, nargs='+', help='List of ingredients to use for finding the closest cuisines.')”

This line defines another command-line argument named --ingredient that expects a list of strings as input, and uses the nargs='+' option to allow the user to provide one or more ingredients when running the script.

The main() function is called when the script is run from the command line.
## Test Functions:
### test_load_data: 
This process tests the load_data method, which retrieves a list of dictionaries after reading data from a JSON file. The function produces a temporary JSON file with test data, uses the path to this file to execute the load_data method, and then verifies that the data returned matches what was intended.
### test_preprocess_data: 
This function evaluates the preprocess_data function, which prepares data for vectorization by taking a list of dictionaries containing the names of dishes and ingredient lists. The preprocess_data function is called using a test input list of dictionaries, and the resulting data is compared to the expected data.
### test_train_model:
This function tests the NearestNeighbors class from scikit-learn's NearestNeighbors function, which trains a k-nearest neighbors model. In order to determine whether the returned NearestNeighbors instance contains the expected parameters, the function runs the train_model function with a test input matrix and additional arguments.
### test_find_closest_cuisines:
This function evaluates the find_closest_cuisines function, which locates the k closest cuisines to a list of input ingredients using a trained k-nearest neighbors model and a vectorizer. With a test input model, vectorizer, and ingredients list, the function calls the find_closest_cuisines function and tests to see if the closest cuisines list it returns matches the expected format and values.
## Bugs and Assumptions:
Without looking for duplication, the preprocessing function lowercases all ingredient names. When various substances have the same name but distinct situations, this could result in information being lost.

It's possible that not all sorts of components can be represented numerically using the count vectorizer that was used to translate the list of ingredients. For instance, the count vectorizer might not be able to accurately collect ingredients that are represented by phrases or words.

The program assumes that the cuisine of a dish is solely determined by the ingredients used and not influenced by other factors such as cooking methods or cultural traditions.

The program assumes that the closest cuisines are the most similar to the input ingredients based on the chosen similarity metric and k value.
