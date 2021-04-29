##!/usr/bin/env python
# coding: utf-8

# # Recommender System: Final Project ME396P
# ## Initialize

import pickle
import pandas as pd
from surprise import SVD


# Load Pickle model

cosine_sim = pickle.load(open('../Serialized_objects/cosine_sim.sav', 'rb'))

# Load processed data frame

dfMovies = pd.read_csv('../Datasets/dfMovies.csv')
dfMoviesGenre = pd.read_csv('../Datasets/dfMoviesGenre.csv')
dfRatUnique = pd.read_csv('../Datasets/dfRatUnique.csv')

# Let us create an original_title-indexed relation

titles = dfMovies['title']

indices = pd.Series(dfMovies.index,index=dfMovies['title'])
indices

# Function for the API to call easily

def get_recommendations_description(title):
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

# Get the 10 movies with their overview most similar to Toy Story

# print(get_recommendations_description('Toy Story'))

# Load Second Pickle model

cosine_simOT = pickle.load(open('../Serialized_objects/cosine_simOT.sav', 'rb'))

# Let us create an original_title-indexed relation

titlesOT = dfMoviesGenre['original_title']

indicesOT = pd.Series(dfMoviesGenre.index,index=dfMoviesGenre['original_title'])
indicesOT

# Function for the API to call easily

def get_recommendations_plot_cast(title):
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

# Get the 10 movies with their genre and cast most similar to The Dark Knight

# print(get_recommendations_plot_cast('The Dark Knight'))

# Load Third Pickle model

mdlSvdMvsRtg = pickle.load(open('../Serialized_objects/svd_model.sav', 'rb'))

# Function to predict the score of a specific movie from a user

def get_ind_score(uid, movieIdx):
    pred = mdlSvdMvsRtg.predict(uid, movieIdx)
    if pred.details['was_impossible'] == False:
        return round(pred.est,2)
    else:
        return -1

# Function for the API to call easily

def get_recommendations_per_user(uid, n=10):
    # Apply prediction algorithm for that specific user rating
    dfRatUnique['predicted_rating'] = dfRatUnique.apply(lambda x: get_ind_score(uid, x['id']), axis=1)
    # Sort by descending value
    dfRatUnique.sort_values('predicted_rating', ascending=False, inplace=True)
    # Return first N elements
    return (dfRatUnique['original_title'].tolist())[:n]

# Function to get the list of users IDs
def get_user_IDs():
  return dfRatUnique['userId'].tolist()

# Give me the top 10 movies the user will probably like

# print(get_recommendations_per_user(1))


