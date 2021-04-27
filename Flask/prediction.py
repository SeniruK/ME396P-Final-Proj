import pickle
import pandas as pd

# In[38]:


# Deserialize objects

cosine_sim = pickle.load(open('../Serialized_objects/model.pkl', 'rb'))
dfMovies = pickle.load(open('../Serialized_objects/dfMovies.pkl', 'rb'))

# Let us create a title-indexed relation

titles = dfMovies['title']

indices = pd.Series(dfMovies.index,index=dfMovies['title'])
indices

# In[39]:

def get_recommendations(title):

	# Check if the movie is included in the data set and it is valid (exists)
	try:
		# Retrieve the movie index by title
		idx = indices[title]
	except:
		return Exception('Movie not found') # Return error

	# Retrieve those movies with similarity to whatever passed
	sim_scores = list(enumerate(cosine_sim[idx]))
	# Sort by score, descending order
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	# Keep the top 10 most similar movies
	sim_scores = sim_scores[1:11]
	# Return results in df form
	movie_indices = [i[0] for i in sim_scores]
	return titles.iloc[movie_indices]

# In[44]:

# print(get_recommendations('Toy Story'))

# .to_list() for list format


# ## TODO: Collaborative Filtering Recommendation

# In[41]:


# Serialize model
# pickle.dump()


# ### Notebook developed by Alejandro Gleason Méndez,  Emily Crowell, Ricardo Antonio Vázquez Rodríguez, Seniru Kottegoda
