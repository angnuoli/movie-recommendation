import numpy as np
from sklearn.model_selection import train_test_split

from data_structure.StaticData import StaticData
from utils.Vectorizer import MyVectorizer


def str_to_vector(text=""):
    if len(text) == 0:
        print("The length is not valid.")
        return

    words = str_to_words(text)
    return list_words_to_vector(words)


def str_to_words(text=""):
    vectorizer = MyVectorizer()
    words = vectorizer.transform(text)
    return words


def split_dataset(dataset, test_size=0.2, train_size=0.7):
    """
    This method is to split the dataset into train dataset and test dataset.
    :param train_size:
    :param test_size:
    :param dataset:
    :return:
    """
    # train, test = train_test_split(dataset, test_size)
    pass
    # return train, test


def list_words_to_vector(words=None):
    """
    Convert a list of words to a ndarray(300) vector.

    :param words:
    :return: vector: ndarray(300)
    """
    if words is None:
        raise RuntimeError("The words list is None")
    m = len(words)
    vector = np.zeros(300)
    word_vectors = StaticData.word_vectors
    for word in words:
        if word in word_vectors.vocab:
            vector += StaticData.word_vectors.get_vector(word)

    return vector / m


def transform(users):
    temp_users = {}
    for id in users.keys():
        temp_users[id] = users[id].reviews

    return temp_users