import gensim
import  numpy as np

sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)

try:
    print(model['12'])
except KeyError as e:
    # print(e)
    print(model.seeded_vector('first'))