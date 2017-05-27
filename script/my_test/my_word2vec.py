from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

word_vectors = KeyedVectors.load_word2vec_format('../data/cn.skipgram.bin/cn.skipgram.bin', binary=True,
                                                 unicode_errors='ignore')
# 距离，越大表示越不相似
print(word_vectors.wmdistance(['中国', '打败', '美国'], ['游戏', '好玩']))
print(word_vectors.wmdistance(['游戏', '好玩'], ['游戏', '好玩']))
print(word_vectors.wmdistance(['中国', '打败', '美国'], ['中国', '打败', '美国']))
print(word_vectors.wmdistance(['中国', '打败', '美国'], ['美国', '中国', '打败']))

