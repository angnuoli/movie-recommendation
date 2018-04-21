class User():

    def __init__(self, id):
    	# reviews ndarray, time increasing
        self.reviews = []
        self.id = id


class Movie():

    def __init__(self, id):
        self.id = id
        self.review = []
