#!/usr/bin/env python
# coding: utf-8

# # Recommender System: Final Project ME396P
# ## Initialize

# In[42]:


import numpy as np                 # Used for array management
import pandas as pd                # Used for data management
import missingno as msno           # Used for visualizing missing values
import matplotlib.pyplot as plt    # Used for data visualization


# ## Load Data

# When handling large datasets, it is common practice to load the dataset (usually in .csv form) into a Pandas DataFrame.
# 
# A DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. It is very similar to a spreadsheet. In our code, all data frames are denoted with "df" in front of the data frame name.

# In[43]:


# Load "credits" dataset into a dataframe
# This dataset contains the credits for each movie, both cast and crew, along with the movie's ID number.
# Each ID number is unique to every movie, which is used to identify movies across different datasets

dfCredits = pd.read_csv('credits.csv', encoding='utf-8', error_bad_lines=False)
dfCredits.head()


# In[44]:


# Load "keywords" dataframe
# This dataset contains keywords for each movie in the form of a dictionary

dfKeywords = pd.read_csv('keywords.csv')
dfKeywords.head()


# In[45]:


# Load "links small" dataframe
# This is a subset of a larger "links" dataset
# This dataset indexes each movie and includes the ID number 

dfLinks = pd.read_csv('links_small.csv')
dfLinks.head()


# In[46]:


# Load "movies metadata" dataframe
# This dataset contains the data of each movie, including title, overview, language, runtime, ID number, vote avg, and vote count.

dfMovies = pd.read_csv('movies_metadata.csv')
dfMovies.head()


# In[47]:


# Remove unformatted IDs
# Formatted IDs are strings comprised of just integers

dfMovies = dfMovies[dfMovies['id'].apply(lambda x: str(x).isdigit())]

dfMovies[['id']] = dfMovies[['id']].apply(pd.to_numeric)


# In[48]:


# Load "ratings small" dataframe
# This dataset is a subset of a larger user ratings dataset. Each user was asked to rate some of their favorite movies along with a few least favorite movies

dfRatings = pd.read_csv('ratings_small.csv')
dfRatings.head()


# In[49]:


# Join the movies with the ratings dataset

dfRatings = dfRatings.rename(columns={'movieId': 'id'})
dfRatings = dfRatings.merge(dfMovies,on='id')

# Keep only useful columns
dfRatings = dfRatings[['userId', 'id', 'rating', 'original_title']]
dfRatings.head()


# In[50]:


# Visualize missing values as a matrix. The white spaces indicate a missing value.

msno.matrix(dfMovies)


# It looks like the "belongs_to_collection", "homepage", and "tagline" columns have several missing values. It is best to not include these in our recommendation models.

# ## Content Based Filtering

# ### Browse similar content - Plot based

# In[51]:


# Let's drop columns that are almost empty, and rows that don't have neither description AND tagline

dfMovies = dfMovies.drop(['belongs_to_collection', 'homepage'], axis=1)

dfMovies = dfMovies.dropna(subset=['overview', 'tagline'], how='all')

dfMovies.shape


# In[52]:


# Removing duplicates by title, keeping the first appearance

dfMovies.drop_duplicates(subset='title', keep='first', inplace=True)

dfMovies.shape


# In[53]:


# Prepare description column

dfMovies['overview'] = dfMovies['overview'].fillna('')

dfMovies['tagline'] = dfMovies['tagline'].fillna('')

dfMovies['description'] = dfMovies['overview'] + ' ' + dfMovies['tagline']

dfMovies['description'][1]


# In[54]:


import re
import nltk
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

def clean_document(row):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(row))
    
    # Remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    return document


# In[55]:


# This is the clean description of the movie in index 1.
# Everything is lowercase without any special characters.
# This makes comparing movie descriptions very easy.
dfMovies['clean_description'] = dfMovies['description'].apply(clean_document)
dfMovies['clean_description'][1]


# In[56]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Remove all english stop words such as 'the', 'a', 'and'
mdlTfvMvs = TfidfVectorizer(stop_words='english')

tfidf_matrix = mdlTfvMvs.fit_transform(dfMovies['clean_description'])

tfidf_matrix.shape


# In[57]:


# There are 41308 movies and 68533 different words


# For our content-based movie description recommmender, we used the linear_kernel function from scikit-learn. The user will enter a movie and then the recommender will output the top ten most related movies.
# 
# This function creates a document-term matrix to find which movie descriptions are most related to the entered movie. A document-term matrix collects all the words in each movie description and finds the frequency of each word. Then, when compared to the words in the descriptions of the movie that the user entered, the matrix will find which other movie decriptions have a similar frequency of words.
# 
# From here, we can rank which movies have the higher frequency of similar words.

# ![Screen%20Shot%202021-04-28%20at%209.11.40%20PM.png](attachment:Screen%20Shot%202021-04-28%20at%209.11.40%20PM.png)

# Fig 1. Model of a document-term matrix. For this recommender, the "D" represents the movie descriptions, and the "T" represents each word in the entered movie description.

# In[16]:


# We could use the cosine_similarity() method, but we'll rather use linear_kernel for quickness

from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim.shape


# In[36]:


# Export to Pickle

import pickle

filename = 'cosine_sim.sav'
pickle.dump(cosine_sim, open(filename, 'wb'))


# In[17]:


# Let us create a title-indexed relation

titles = dfMovies['title']

indices = pd.Series(dfMovies.index,index=dfMovies['title'])
indices


# In[18]:


# Function for the API to call easily

def get_recommendations_description(title):
    try:
        # Retrieve the movie index by title
        idx = indices[title] 
        # Retrieve those movies with similarity to whatever passed
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Sort by score, descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Keep the top 10 most similar movies
        sim_scores = sim_scores[1:11]
        # Return results in df form
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]
    except:
        raise ValueError("The movie entered is not part of the dataset")


# In[19]:


# Get the 10 movies with their overview most similar to Toy Story
get_recommendations_description('Toy Story')


# This recommender function just listed the top ten movies with the most similar descriptions as Toy Story.
# 
# The most movies at the top of the list are, of course, the Toy Story sequels. The rest of the movies have similar plots to Toy Story based on the movie descriptions.

# ### Browse similar content - Cast and Keywords based
# In this instance, our goal is to have a new dataframe that has title, cast, director, keywords and genres! This will group movies by their producers, since the director tends to use the same cast, but we will also be using keywords and genres to recommend similar movies (plot based).

# In[60]:


# Join the movies with the credits dataset
dfMovies = dfMovies.merge(dfCredits, on='id')

# Join the movies with the keywords dataset
dfMovies = dfMovies.merge(dfKeywords, on='id')
dfMovies.head()


# In[61]:


# Create new df that holds the 3 to be used columns (cast, keywords and genres)

dfPrdCmp = dfMovies[['id', 'original_title', 'genres', 'cast', 'crew', 'keywords']]
dfPrdCmp.head()


# In[62]:


# This library helps us parse stringified objects to python objects

from ast import literal_eval

for col in ['genres', 'cast', 'crew', 'keywords']:
    dfPrdCmp[col] = dfPrdCmp[col].apply(literal_eval)


# In[63]:


# Function to convert a list to a space separated string
def listToString(s):
    str1 = " "
    return str1.join(s)

# Function to retrieve 'Director' from an object
def get_director(row):
    for x in row:
        if x['job'] == 'Director':
            return x['name']
    # Return empty string if director was not found
    return ""

# Function to get the first three names from an object
def get_list_format(row):
    elements = []
    if isinstance(row, list):
        for element in row:
            elements.append(element['name'])
        # Keep only 3 elements
        if len(elements) > 3:
            elements = elements[:3]
        return elements

    # Return empty list else
    return elements


# In[64]:


# Retrieve directors and remove crew column

dfPrdCmp['director'] = dfPrdCmp.apply(lambda x: get_director(x['crew']),  axis=1)

dfPrdCmp = dfPrdCmp.drop(['crew'], axis=1)

dfPrdCmp.head()


# In[65]:


# Retrieve and format keywords, cast and genres

for col in ['genres', 'cast', 'keywords']:
    dfPrdCmp[col] = dfPrdCmp[col].apply(get_list_format)
    dfPrdCmp[col] = dfPrdCmp[col].apply(listToString)
    
dfPrdCmp.head()


# In[66]:


# Define the 'Document' column

dfPrdCmp['cast_plot'] = dfPrdCmp['genres'] + " " + dfPrdCmp['cast'] + " " + dfPrdCmp['director'] + " " + dfPrdCmp['keywords']
dfPrdCmp['cast_plot'][1]


# In[67]:


# Clean the 'Document' column

dfPrdCmp['cast_plot_clean'] = dfPrdCmp['cast_plot'].apply(clean_document)
dfPrdCmp['cast_plot_clean'][1]


# In[68]:


# Remove all english stop words such as 'the', 'a', 'and'

mdlTfvMvs = TfidfVectorizer(stop_words='english')

tfidf_matrix = mdlTfvMvs.fit_transform(dfPrdCmp['cast_plot_clean'])

tfidf_matrix.shape

# There are 42308 movies and 78477 different words


# In[69]:


# Here we will use the cosine similary method just for educational purposes

# from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix
cosine_simOT = linear_kernel(tfidf_matrix, tfidf_matrix) #, dense_output=True)
cosine_simOT.shape


# In[71]:


# Export to Pickle

filename = 'cosine_simOT.sav'
pickle.dump(cosine_simOT, open(filename, 'wb'))


# In[72]:


# Let us create an original_title-indexed relation

titlesOT = dfPrdCmp['original_title']

indicesOT = pd.Series(dfPrdCmp.index,index=dfPrdCmp['original_title'])
indicesOT


# In[81]:


# Function for the API to call easily

def get_recommendations_plot_cast(title):
    try:
        # Retrieve the movie index by title
        idx = indicesOT[title] 
        # Retrieve those movies with similarity to whatever passed
        sim_scores = list(enumerate(cosine_simOT[idx]))
        # Sort by score, descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Keep the top 10 most similar movies
        sim_scores = sim_scores[1:11]
        # Return results in df form
        movie_indices = [i[0] for i in sim_scores]
        return titlesOT.iloc[movie_indices]
    except:
        raise ValueError("The movie entered is not part of the dataset")


# In[32]:


# Get the 10 movies with their genre and cast most similar to The Dark Knight
get_recommendations_plot_cast('Toy Story')


# Here, we have the top ten movies with the most similar cast and keywords. This list can pertain more to genres or select demographics, as we can see that these movie reommendations are more kid-friendly.
# 
# Let's compare these movie recommendations to the movie description-based recommender:

# In[33]:


get_recommendations_description("Toy Story")


# The first few movies from both recommendation lists are similar, but they differ as we go down the list.
# 
# This demonstrates how the recommender you choose will affect the recommendations.

# ## Collaborative Filtering Recommendation

# Collaborative filtering finds recommendations for users based on other users' selections.
# 
# In our "ratings" dataset, several users rated their favorite movies along with some of their least favorite movies. Collaborative filtering will compare all the ratings from all the users, and then make recommendations for users who have similar movie ratings.
# 
# For our code, we will create a function that can predict each user's ratings for each movie, which will be implemented to our collaborative filtering recommender that will recommend the top ten movies with the highest predicted movie ratings for that user.

# ![Screen%20Shot%202021-04-28%20at%2010.04.58%20PM.png](attachment:Screen%20Shot%202021-04-28%20at%2010.04.58%20PM.png)

# Fig 2. Model of a collaborative filtering network. This implements the ratings of others will similar movie tastes and makes recommendations for users.

# In[73]:


# Rating scale goes from 0 to 5

dfRatings.describe()


# In[74]:


# Retrieve the users for login information

print("The list of existing users is:", set(dfRatings['userId'])) 


# In[76]:


# !pip install scikit-surprise


# In[36]:


from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split

# Specify what the rating scale was
reader = Reader(rating_scale=(0, 5))

# The columns must correspond to user id, item id and ratings
X = Dataset.load_from_df(dfRatings[['userId', 'id', 'rating']], reader)

# Sample random trainset and testset
X_train, X_test = train_test_split(X, test_size=.25)


# In[50]:


from surprise import SVD
from surprise import accuracy

# Define SVD model
mdlSvdMvsRtg = SVD()

# Fit SVD model
mdlSvdMvsRtg.fit(X_train)

# Predict ratings for testset
test_pred = mdlSvdMvsRtg.test(X_test)

# Evaluate SVD model
accuracy.rmse(test_pred) # this means that our data will be at most .90 units from the model


# In[ ]:


# Export SVD model to pikle

filename = 'svd_model.sav'
pickle.dump(mdlSvdMvsRtg, open(filename, 'wb'))


# This error shows that the mean square error for estimating each user's movie rating for a select movie. Our predicted movie rating, which is on a scale of 1-5, has a mean square error of 0.90.

# In[38]:


# Cross validate our potential models

from surprise.model_selection import cross_validate

cross_validate(mdlSvdMvsRtg, X, cv=5)


# In[39]:


# Score new data points

movieIdx = 350
pred = mdlSvdMvsRtg.predict(1, movieIdx) 
print("Have user", pred.uid, "watched movie with index", movieIdx, "the predicted score would be", 
      (str(round(pred.est, 2)) + "."), "Was the estimation impossible to do?", pred.details['was_impossible'])


# In[77]:


# Create dataframe to hold movies in the ratings dataframe

dfRatUnique = dfRatings.drop_duplicates(subset='id', keep="first")

dfRatUnique["predicted_rating"] = ""

dfRatUnique.head()


# In[78]:


dfRatUnique.to_csv('dfRatUnique.csv')


# In[38]:


# Function to predict the score of a specific movie from a user

def get_ind_score(uid, movieIdx):
    try:
        pred = mdlSvdMvsRtg.predict(uid, movieIdx)
        if pred.details['was_impossible'] == False:
            return round(pred.est,2)
        else:
            return -1
    except:
        raise ValueError("The user ID or movie ID entered is not included in the database")


# In[40]:


# Function for the API to call easily

def get_recommendations_per_user(uid, n=10):
    try:
        # Apply prediction algorithm for that specific user rating
        dfRatUnique['predicted_rating'] = dfRatUnique.apply(lambda x: get_ind_score(uid, x['id']), axis=1)
        # Sort by descending value
        dfRatUnique.sort_values('predicted_rating', ascending=False, inplace=True)
        # Return first N elements
        return (dfRatUnique['original_title'].tolist())[:n]
    except:
        raise ValueError("The user ID entered is not included in the database")


# In[41]:


# Give me the top 10 movies the user will probably like
get_recommendations_per_user(1)


# Here, we have the top ten movie recommendations for user 1. The "get_ind_score" function predicted the movie ratings that user 1 would give for each movie, and then compiled the top ten movie ratings into a list.

# ### Notebook developed by Alejandro Gleason Méndez,  Emily Crowell, Ricardo Antonio Vázquez Rodríguez, Seniru Kottegoda

# # APPENDIX

# ## How to Run Movie Finder:

# ### 1. Add dataset .csv files:

# a. Create a free Kaggle account: https://www.kaggle.com
# 
# b. Download the dataset from Kaggle: https://www.kaggle.com/rounakbanik/the-movies-dataset/download
# 
# c. Move all the .csv files inside the Datasets folder

# ### 2. Install the needed modules

# a. Run “pip3 install surprise” or "conda install -c conda-forge scikit-surprise", depending on your system

# ### 3. Run the model creation file inside the repository at Jupyter/movieRecomendationModel.py

# a. Previous requirements: 13 GB+ free space available in order to save the model file.
# 
# b. In order to do this execute the following command inside the Jupyter folder: python movieRecomendationModel.py
# 
# c. This should take some minutes to finish running. (about 15 minutes)
# 
# d. Once it finishes running, you should see files have been exported to the folder Serialized_objects

# ### 4. Run the application

# a. Go to the Flask folder.
# 
# b. Run the app.py (python app.py)

# ## Testing and Troubleshooting

# ### Testing the content-based movie description filter recommenders with correct and incorrect inputs

# Let's enter a movie in the database. We should get a list of the top ten most related movies to the movie input

# In[91]:


get_recommendations_description("The Dark Knight")


# As we can see, our function worked properly.
# 
# Now, let's misspell the movie, which will not be in the movie database.

# In[92]:


# Let's take out the word "The", which is not the correct title:

get_recommendations_description("Dark Knight")


# Here, we can see that the function raised the ValueError with the message "The movie entered is not part of the dataset." This is the intended response. The user will have to enter the correct movie spelling.

# Now, let's test the other filter functions.

# ### Testing the content-based cast and keyword filter recommenders with correct and incorrect inputs

# The correct response:

# In[79]:


get_recommendations_plot_cast("The Dark Knight")


# The incorrect response:

# In[90]:


get_recommendations_plot_cast("Dark Knight")


# Here, we can see the raised ValueError.

# ## References

# Banik, R. (2017, November 10). The Movies Dataset. Retrieved from https://www.kaggle.com/rounakbanik/the-movies-dataset

# Maklin, C. (2019, July 21). TF IDF: TFIDF Python Example. Retrieved from https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
# 

# Text Mining: Word Frequency Models: NLP Blog Post. (2019, July 03). Retrieved from https://www.mosaicdatascience.com/2015/10/12/text-mining-word-frequency-models/
# 
