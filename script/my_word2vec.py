from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

# word_vectors = KeyedVectors.load_word2vec_format('../data/cn.skipgram.bin/cn.skipgram.bin', binary=True,
#                                                  unicode_errors='ignore')
# print(type(word_vectors.syn0))
# print(np.shape(word_vectors.syn0))
# print(word_vectors.word_vec('中国', use_norm=True))
# 距离，越大表示越不相似
# print(word_vectors.wmdistance(['中国', '打败', '美国'], ['游戏', '好玩']))
# print(word_vectors.wmdistance(['中国', '打败', '美国'], ['美国', '中国', '击败']))

sentences = [['中国', '打败', '美国'], ['游戏', '好玩'], ['美国', '中国', '击败']]
model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
print(model.wv.syn0norm)
print(model.wv.word_vec('中国',use_norm=True))
