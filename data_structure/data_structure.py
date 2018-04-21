class User:

    def __init__(self, id_):
        # reviews ndarray, time increasing
        self.reviews = []
        self.id = id_


class Movie:

    def __init__(self, id_):
        # reviews ndarray, time increasing
        self.id = id_
        self.reviews = []
