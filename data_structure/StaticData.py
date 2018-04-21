# -*- coding: utf-8 -*-

import gensim
from gensim.models import KeyedVectors

class StaticData:

    word_vectors = KeyedVectors.load_word2vec_format('../dataset/GoogleNews-vectors-negative300.bin', binary=True)

    # database
    users = {}
    movies = {}

    # raw reviews
    reviews = []
    train_reviews = []
    test_reviews = []