import numpy as np

from data_structure import StaticData


def str_to_vector(text=""):
    if len(text) == 0:
        print("The length is not valid.")
        return

    words = str_to_words(text)
    return list_words_to_vector(words)

def str_to_words(text = ""):
	identify = string.maketrans('', '')  
	delEStr = string.punctuation +'0123456789'
	cleanLine =text.translate(identify,delEStr) 
	words=[i for i in cleanLine.split(' ')if i!='']
	return words



def list_words_to_vector(words=[]):
    """
    Convert a list of words to a ndarray(300) vector.

    :param words:
    :return: vector: ndarray(300)
    """
    m = len(words)
    vector = np.zeros(300)
    for word in words:
        vector += StaticData.word_vectors.get_vector(word)

    return vector / m


