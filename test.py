import gensim
from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('dataset/GoogleNews-vectors-negative300.bin', binary=True)
# if you vector file is in binary format, change to binary=True

word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])