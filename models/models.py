"""
Contains various recommondation implementations. 

All classes should be derived from the BaseRecommender
"""

import numpy as np
import pickle
import pandas as pd
from movierecommender.datasets import movies, ratings

import tensorflow.keras as keras
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process





class BaseRecommender:
    """
    Recommender interface class
    """

    def __init__(self, user_item_mat):     
        self.user_item_mat = user_item_mat  


    def fit(self):
        """Fit a model to the user item matrix
        """
        raise NotImplementedError



    def recommend(self, user_id, k):
        """
        Give a list of top-k recommondations for a user in the dataset
        """
        raise NotImplementedError



class MyFancyModelRecommender:

    def __init__(self, model_file='model_NCF.h5'):
        self.model = load_model("model_NCF.h5")


    def user_recommendation_internal(self, user_number, number_of_recommendation=5):
        # array for movies
        m_id =list(ratings.movieId.unique())   # list of all unique movie id's
        movie_arr = np.array(m_id) #get all movie IDs
        item_enc = LabelEncoder()
        movie_arr_en = item_enc.fit_transform(movie_arr) # transfor upto number of unique movie id's    
        #array for user
        user = np.array([user_number for i in range(len(m_id))])
        # make a prediction using the model
        pred = self.model.predict([movie_arr_en, user])
        #reshape to single dimension
        pred = pred.reshape(-1) 
        #dictionary of all movie_id'ss and movie_titles
        movie_id_dict = dict(zip(movies['movieId'], movies['title']))
        #create a dataframe with predicted ratings
        recommendation = pd.DataFrame({'predicted_rating': pred, 'movieId': ratings.movieId.unique()}).sort_values(by=['predicted_rating'], ascending=False)
        #mapping movie titles with movie_id's
        movies_ = recommendation['movieId'].map(movie_id_dict)
        #make a prediction of top 
        movies_ = movies_.head(number_of_recommendation)
        #make a list
        movies_list = movies_.tolist()
        return movies_list    
    
    # def convert_user_input(self, user_input_movies):  
    #     #Get the indexes of movies entered by user
    #     user_movie_index = [process.extractOne(movie, movies['title'])[2] for movie in user_input_movies]    
        
    # # if len(user_movie_index) != len(user_input_movies):
    # #     unknown = len(user_input_movies) - len(user_movie_index)
    # #     print(f'I could not find {unknown} movie(s) from your list')
    #     return user_movie_index

    def user_recommendation_external(self, user_input_movies, number_of_recommendation=5):
        #Get the indexes of movies entered by user
        user_mov_index = [process.extractOne(movie, movies['title'])[2] for movie in user_input_movies]
        # array for movies
        m_id =list(ratings.movieId.unique())   # list of all unique movie id's
        interacted_items = user_mov_index
        not_interacted_items = set(m_id) - set(interacted_items)
        selected_not_interacted = np.random.choice(list(not_interacted_items), len(m_id)-len(user_mov_index))
        movie_arr = np.concatenate((np.array(interacted_items), selected_not_interacted), axis=None)
        item_enc = LabelEncoder()
        movie_arr_en = item_enc.fit_transform(movie_arr) # transfor upto number of unique movie id's
        #array for user 
        user = np.array([1 for i in range(movie_arr_en.shape[0])])
        user_enc = LabelEncoder()
        user_en = user_enc.fit_transform(user)
        # make a prediction using the model
        pred = self.model.predict([movie_arr_en, user])
        #reshape to single dimension
        pred = pred.reshape(-1) 
        #dictionary of all movie_id's and movie_titles
        movie_id_dict = dict(zip(movies['movieId'], movies['title']))
        #create a dataframe with predicted ratings
        recommendation = pd.DataFrame({'predicted_rating': pred, 'movieId': ratings.movieId.unique()}).sort_values(by=['predicted_rating'], ascending=False)
        #mapping movie titles with movie_id's
        movies_ = recommendation['movieId'].map(movie_id_dict)
        #make a prediction of top 
        movies_ = movies_.head(number_of_recommendation)
        #make a list
        movies_list = movies_.tolist()
        return movies_list

# class GlobalRecommender:

#     def __init__(self, model_file='./trained_models/my_model.pickle'):
#         with open(model_file, 'rb') as f:
#             self.model = pickle.load(f)


#     def recommend(self, list_of_movies):
#         # preprocess the data (imputing, scaling)
#         # make rating prediction
#         # self.model.predict()
#         # filter for unseen movies
#         # return the top 5        
#         return list_of_movie_ids
