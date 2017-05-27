import numpy as np
import tensorflow
from keras import Input, optimizers
from keras import backend as K
from keras.engine import Model
from keras import layers
from keras.layers import Bidirectional, LSTM, merge, Reshape, Lambda, Dense, BatchNormalization

K.clear_session()
print("设置显卡信息...")
# 设置tendorflow对显存使用按需增长
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)

question_max_len = 40
answer_max_len = 40
embedding_dim = 300
input_question = Input(shape=(question_max_len, embedding_dim))
input_answer = Input(shape=(answer_max_len, embedding_dim))

# 双向lstm
question_lstm = Bidirectional(LSTM(64))
answer_lstm = Bidirectional(LSTM(64))

encoded_question = question_lstm(input_question)
encoded_answer = answer_lstm(input_answer)

cos_distance = merge([encoded_question, encoded_answer], mode='cos', dot_axes=1)
cos_distance = Reshape((1,))(cos_distance)
cos_similarity = Lambda(lambda x: 1 - x)(cos_distance)

predictions = Dense(1, activation='sigmoid')(cos_similarity)

model = Model([input_question, input_answer], [predictions])
sgd = optimizers.SGD(lr=0.1, clipvalue=0.5)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
model.summary()

# 下面是训练
questions = np.load('train' + '_' + 'questions' + '.npy')
answers = np.load('train' + '_' + 'answers' + '.npy')
labels = np.load('train' + '_' + 'labels' + '.npy')

# 下面是 dev 验证
dev_questions = np.load('dev' + '_' + 'questions' + '.npy')
dev_answers = np.load('dev' + '_' + 'answers' + '.npy')
dev_labels = np.load('dev' + '_' + 'labels' + '.npy')

# 开始迭代
model.fit([questions, answers], [labels],
          epochs=2,
          batch_size=256,
          validation_data=([dev_questions, dev_answers], [dev_labels]))
# 预测
print('开始预测！')
predict = model.predict([dev_questions, dev_answers], verbose=1, batch_size=256)
print(predict)
np.save('predict.npy', predict)
