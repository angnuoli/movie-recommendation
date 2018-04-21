# -*- coding: utf-8 -*-

import json
import codecs

from data_structure import StaticData
from data_structure.data_structure import User
from utils.utils import str_to_vector


def split_reviews():
    train_reviews = []
    test_reviews = []

    return train_reviews, test_reviews


def preprocess():
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

    for train_review in train_reviews:
        review = "{} {}".format(train_review['review'], train_review['title'])
        movie_id = train_review['movie']
        user_id = train_review['user']
        rating = train_review['rating']
        vector = str_to_vector(review)

        if user_id not in users.keys():
            users[user_id] = User(user_id)


    print(reviews[0]['review'])
