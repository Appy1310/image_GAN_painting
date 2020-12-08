from flask import Flask, request, render_template
# from movierecommender import datasets
# from movierecommender import utils

# from movierecommender.models import MyFancyModelRecommender
import random
import os
import sys
#from os import listdir
from random import random
import numpy as np
from numpy import load, zeros, ones, asarray
from numpy.random import randint
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims

import matplotlib.pyplot as plt
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot

import os
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory

from prediction import load_image, paint_generator

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
#my_model = MyFancyModelRecommender()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load the models
cust = {'InstanceNormalization': InstanceNormalization}
# from vangogh to photo
model_vangogh = load_model('model_photo_to_vangogh.h5', cust, compile=False)
model_vangogh.compile(
    loss=[
        'mse', 'mae', 'mae', 'mae'], loss_weights=[
            1, 5, 10, 10], optimizer=keras.optimizers.Adam(
                lr=0.0002, beta_1=0.5))
# from photo to vangogh
model_monet = load_model('model_photo_to_monet.h5', cust, compile=False)
model_monet.compile(
    loss=[
        'mse', 'mae', 'mae', 'mae'], loss_weights=[
            1, 5, 10, 10], optimizer=keras.optimizers.Adam(
                lr=0.0002, beta_1=0.5))


# start page
@app.route('/')
def index():
    # print('oh!')
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        print(dict(request.args))
        # print(dict(request.values)['artists'])
        file = request.files['file']  # get the input image file
        # get the input of desired artist
        artist = dict(request.values)['artists']

        if artist == 'vincent-van-gogh':
            image = load_image(file)
            image_generated = paint_generator(
                model_vangogh, image, 'generated_' + file.filename)
            image_file_path = f'/static/img/generated_{file.filename}'
        if artist == 'claude-monet':
            image = load_image(file)
            image_generated = paint_generator(
                model_monet, image, 'generated_' + file.filename)
            image_file_path = f'/static/img/generated_{file.filename}'
        # if user does not select file, browser also
        # submit an empty part without filename
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #     return redirect(url_for('uploaded_file',
        #                             filename=filename))
        return render_template(
            'results.html',
            image_file_path=image_file_path,
            artist=artist)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

# # recommender result page
# @app.route('/movie/recommender-inside')
# def recommender_inside():
#     # how to access the query in the url?!
#     dict_user = dict(request.args)

#     user_number= int(list(dict_user.values())[0])

#     print(user_number)
#     print(type(user_number))
#     recommendation_list = my_model.user_recommendation_internal(user_number)

#     # search_results = utils.search_movie(query, datasets.movies)
# return render_template('results.html', user_number = user_number,
# recommendation_list = recommendation_list)


# # recommender result page
# @app.route('/movie/recommender-outside')
# def recommender():
#     # how to access the query in the url?!
#     dict_list = dict(request.args)
#     list_of_movies = list(dict_list.values())
#    # print(list_of_movies)
#     recommendation_list = my_model.user_recommendation_external(list_of_movies)

#     # search_results = utils.search_movie(query, datasets.movies)
# return render_template('results.html', dict_list = dict_list,
# recommendation_list = recommendation_list)


# # movie info page with a parametrized endpoint
# @app.route('/movie/<int:movie_id>')
# def movie_info(movie_id):
#     # get the title for a specific movie id
#     movie = datasets.movies.loc[movie_id]
#     #return f'movie title: {title}, movie id: {movie_id}'
#     movie_poster = utils.movie_poster_url(movie.title)
#     #print(movie_poster)
# return render_template('movie.html', movie=movie, movie_id=movie_id,
# movie_poster = movie_poster)


if __name__ == "__main__":
    # only run the following lines when you write 'python app.py'
    app.run(debug=True)
