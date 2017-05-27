import numpy as np
import tensorflow
from keras import Input, optimizers
from keras import backend as K
from keras.engine import Model
from keras import layers
from keras.layers import Bidirectional, LSTM, merge, Reshape, Lambda, Dense, BatchNormalization, Activation
from keras.models import Sequential

K.clear_session()
print("设置显卡信息...")
# 设置tendorflow对显存使用按需增长
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)

mask_value = 0
question_max_len = 40
answer_max_len = 40
embedding_dim = 300

model_left = Sequential()
model_left.add(layers.Masking(mask_value=mask_value, input_shape=(question_max_len, embedding_dim)))
model_left.add(layers.BatchNormalization())
model_left.add(layers.Bidirectional(LSTM(32, return_sequences=True)))
model_left.add(layers.Bidirectional(LSTM(32)))


model_right = Sequential()
model_right.add(layers.Masking(mask_value=mask_value, input_shape=(answer_max_len, embedding_dim)))
model_right.add(layers.BatchNormalization())
model_right.add(layers.Bidirectional(LSTM(32, return_sequences=True)))
model_right.add(layers.Bidirectional(LSTM(32)))

model = Sequential()
model.add(layers.Merge([model_left, model_right], mode='cos'))
model.add(layers.Reshape((1,)))
model.add(layers.Lambda(lambda x: 1 - x))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))
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
