import pickle

## Collaborative Filtering Recommendation"""

dfRatings = pickle.load(open('../Serialized_objects/dfRatings.pkl', 'rb'))

# Scale goes from 0 to 5
dfRatings.describe()

#!pip install scikit-surprise

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split

# specify what the rating scale was
reader = Reader(rating_scale=(0, 5))

# the columns must correspond to user id, item id and ratings
X = Dataset.load_from_df(dfRatings[['userId', 'id', 'rating']], reader)

# sample random trainset and testset
X_train, X_test = train_test_split(X, test_size=.25)

from surprise import SVD
from surprise import accuracy

# define SVD model
mdlSvdMvsRtg = SVD()

# fit SVD model
mdlSvdMvsRtg.fit(X_train)

# predict ratings for testset
test_pred = mdlSvdMvsRtg.test(X_test)

# evaluate SVD model
accuracy.rmse(test_pred) # this means that our data will be at most .89 units from the model

# cross validate

from surprise.model_selection import cross_validate

cross_validate(mdlSvdMvsRtg, X, cv=5)

# Score new data points

movieIdx = 350
pred = mdlSvdMvsRtg.predict(1, movieIdx) 
# print("Have user", pred.uid, "watched movie with index", movieIdx, "the predicted score would be", 
#       (str(round(pred.est, 2)) + "."), "Was the estimation impossible to do?", pred.details['was_impossible'])

# Create dataframe to hold movies in the ratings dataframe
dfRatUnique = dfRatings.drop_duplicates(subset='id', keep="first")
dfRatUnique["predicted_rating"] = ""
dfRatUnique.head()

# Function to get top 10 movies for a specific user
def get_recommendations_per_user(uid, n=10):
  # Apply prediction algorithm for that specific user rating
  dfRatUnique['predicted_rating'] = dfRatUnique.apply(lambda x: get_ind_score(uid, x['id']), axis=1)
  # Sort by descending value
  dfRatUnique.sort_values('predicted_rating', ascending=False, inplace=True)
  # Return first N elements
  return (dfRatUnique['original_title'].tolist())[:n]

# Function to predict the score of a specific movie from a user
def get_ind_score(uid, movieIdx):
  pred = mdlSvdMvsRtg.predict(uid, movieIdx)
  if pred.details['was_impossible'] == False:
    return round(pred.est,2)
  else:
    return -1

# Function to get the valid IDs of a user
def get_user_IDs():
  return dfRatings['userId'].tolist()

# params: for user 1, give me their top 10 predicted-rated movies

# print(get_recommendations_per_user(9))
