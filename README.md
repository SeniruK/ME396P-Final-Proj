# Project Description
When individuals have spare time, they usually recur to a form of entertainment, one of the most common forms of entertainment is watching movies, nowadays we can achieve this through different digital services such as Netflix, Hulu, Disney Plus, among many others that offer us a great variety of catalogs and collections. These platforms have grown both technically and in scale and have suggestion algorithms for recommending movies to the users. 

However, as there is a great catalog to browse through, some people take a great amount of time browsing what to watch next. Another issue is that each platform has a different catalog of content and people that have different streaming services need to browse content independently for each platform. Our focus is to be able to suggest movies that a user would enjoy through machine learning and an accessible platform for it.

## Objectives
Design, development and integration of a machine learning based movie recommendation system, that bundles both customer’s behavior and movie similarity to  perform recommendations and predict ratings.

Design and development of an interactive web application for users to use the recommender system, which will be embedded in its backend, for them to get new recommendations based on its profile or other movies preference.

## Requirements
Project proposal presentation can be found on this link.

[Requirements list.](https://docs.google.com/spreadsheets/d/17fanx073Sogh4UoWtw6Q4SxjEr8RLs4qssyUomtI0a8/edit?usp=sharing)

# How to run the code

## Clone the Github Repository
1. Clone this repository into your computer.

## Add Kaggle csv files and dependencies

1. Add Kaggle dataset .csv files:
Download the csv files from the following link. 
Move all of these .csv files inside the Jupyter folder inside the clone repository or in your Colab file tab if you are there.

2. Open and run the Jupyter notebooks cells (FinalNotebook#G07.ipynb) either on Jupyter notebook’s interface or Colab.

3. Install the needed modules
If running on [Colab](http://colab.research.google.com/):
Run the importing cells and if any package is missing (you’ll see an error similar to Python - Module Not Found), just install it by running in a cell the following format -> !pip install package_name
If running on Anaconda, you will need to run first:
conda install -c conda-forge scikit-surprise
conda install -c conda-forge missingno
conda install -c r r-wordnet
Still in Anaconda, if there are any missing packages (Python - Module Not Found), just install them by using the format  pip install package_name on your terminal.
The mandatory modules to install are the following, but you may already have some of them:
Numpy
Pandas
Matplotlib
Pickle
Sklearn
Flask
requests
missingno
nltk

4. Run the Notebook for model creation and csv files.
Previous requirements: 26 GB of free space available in order to save the models file.
Continue running all the cells of the Jupyter Notebook file.
This should take some minutes to finish running (about 15-20 minutes, since each model is being created).
Once it finishes running, you should move the generated csv files (dfMovies.csv, dfMoviesGenre.csv, dfRatUnique.csv) to the Datasets folder, and the Pickle files (cosine_sim.sav, svd_model.sav, cosine_simOT.sav) to the folder Serialized_objects from the cloned repository. In case you don’t want to run the .csv generation cells, you can find them here.

[dfRatings](https://drive.google.com/file/d/1p8QiT9Uc9KMZJQWrjgXx0wcYuwoSkVP-/view?usp=sharing): https://drive.google.com/file/d/1p8QiT9Uc9KMZJQWrjgXx0wcYuwoSkVP-/view?usp=sharing

[dfMovieGenre](https://drive.google.com/file/d/1KU-fk-qeRvf4AD9NnEeux64taHh9yvSw/view?usp=sharing): https://drive.google.com/file/d/1KU-fk-qeRvf4AD9NnEeux64taHh9yvSw/view?usp=sharing

[dfMovies](https://drive.google.com/file/d/1fzK9ZzlrN3AAywh1urPwEgGmydS4rJat/view?usp=sharing): https://drive.google.com/file/d/1fzK9ZzlrN3AAywh1urPwEgGmydS4rJat/view?usp=sharing

5. Run the application
Go to the Flask folder.
Run the app.py (python app.py)
If you want to run the API example as well, run request.py on another terminal.
