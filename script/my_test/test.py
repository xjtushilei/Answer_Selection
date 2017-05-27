import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

texts = ['hello my friend', 'you are beautiful', 'hello are good']  # list of text samples
labels_index = {1, 2}  # dictionary mapping label name to numeric id
labels = [1, 2, 2]  # list of label ids

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print(sequences)

word_index = tokenizer.word_index
for word, i in word_index.items():
    print(word,i)

embeddings_index = {}
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector