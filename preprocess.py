# -*- coding: utf-8 -*-

import json
import codecs

from data_structure import StaticData
from data_structure.data_structure import User, Movie
from utils.utils import str_to_vector


def split_reviews():
    train_reviews = []
    test_reviews = []

    return train_reviews, test_reviews


def preprocess():
    """
    Read data from files. Create users and movies map.

    :return: samples: (user_id, movie_id, rating)
             users: map (user_id -> User)
             movies: map (moview_id -> Movie)
    """
    reviews = []

    with codecs.open('../dataset/data.json', 'r', 'utf-8') as f:
        reviews = json.load(f)

    # data structure: reviews[i]['rating', 'title', 'movie', 'review', 'link', 'user']
    StaticData.reviews = reviews
    train_reviews, test_reviews = split_reviews(reviews)
    StaticData.train_reviews = train_reviews
    StaticData.test_reviews = test_reviews

    users = {}
    movies = {}
    samples = []

    for train_review in train_reviews:
        review = "{} {}".format(train_review['review'], train_review['title'])
        movie_id = train_review['movie']
        user_id = train_review['user']
        rating = train_review['rating']
        vector = str_to_vector(review)

        if user_id not in users.keys():
            users[user_id] = User(user_id)
        if movie_id not in movies.keys():
            movies[movie_id] = Movie(movie_id)

        user = users[user_id]
        movie = movies[movie_id]

        user.reviews.append(vector)
        movie.reviews.append(vector)

        samples.append([user_id, movie_id, rating])

    # print(reviews[0]['review'])
    StaticData.users = users
    StaticData.movies = movies

    return samples, users, movies
