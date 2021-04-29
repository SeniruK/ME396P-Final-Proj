# Final project
Application Programming for Engineers 
Team Repository

# Optimize Prime G07

Team Members:

1. Emily Crowell
2. Seniru Nimeth Kottegoda
3. Ricardo A. Vazquez R.
4. Alejandro Gleason Mendez

# Project Description

When individuals have spare time, they usually recur to a form of entertainment, one of the most common forms of entertainment is watching movies, nowadays we can achieve this through different digital services such as Netflix, Hulu, Disney Plus, among many others that offer us a great variety of catalogs and collections. These platforms have grown both technically and in scale and have suggestion algorithms for recommending movies to the users. 

However, as there is a great catalog to browse through, some people take a great amount of time browsing what to watch next. Another issue is that each platform has a different catalog of content and people that have different streaming services need to browse content independently for each platform. Our focus is to be able to suggest movies that a user would enjoy through machine learning and an accessible platform for it.

# Presentation: 
https://drive.google.com/file/d/1CDpe_ZcExkyv7BjFzCNTFhsWfebe2UYi/view?usp=sharing

# Documentation: 
https://docs.google.com/document/d/1BVOnOAMgsReBJ_D-x-INv5FV7uYFPyeI5qJbYvlor_8/edit?usp=sharing

# How to run it
1. Add dataset .csv files:
  1.1. Create a free Kaggle account: https://www.kaggle.com
  1.2. Download the dataset from Kaggle: https://www.kaggle.com/rounakbanik/the-movies-dataset/download
  1.3. Move all the .csv files inside the Datasets folder

2. Install the needed modules
2.1. Run “pip install surprise”

3. Run the model creation file inside the repository at Jupyter/movieRecomendationModel.py
  3.1. Previous requirements: 13 GB+ free space available in order to save the model file.
  3.2. In order to do this execute the following command inside the Jupyter folder: python movieRecomendationModel.py
  3.3. This should take some minutes to finish running. (about 15 minutes)
  3.4. Once it finishes running, you should see files have been exported to the folder Serialized_objects

4. Run the application
  4.1. Go to the Flask folder.
  4.2. Run the app.py (python app.py)
