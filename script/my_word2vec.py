from gensim.models.keyedvectors import KeyedVectors
import numpy as np

word_vectors = KeyedVectors.load_word2vec_format('../data/cn.skipgram.bin/cn.skipgram.bin',binary=True ,unicode_errors='ignore')
# print(np.shape(word_vectors['你好']))

