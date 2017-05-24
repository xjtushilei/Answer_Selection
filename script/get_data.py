import pickle
import platform
import re
import jieba
import sys, os
from gensim.models.keyedvectors import KeyedVectors as vec
import numpy as np
from keras.utils import np_utils

kind = 'train'
# kind = 'dev'
print("使用数据：", kind)
if 'Windows' in platform.system():
    train_file_dir = 'D:/data/qadata/'
    print("平台为：", platform.system())
if 'Linux' in platform.system():
    train_file_dir = '/home/script/data/qadata/'
    print("平台为：", platform.system())

mask_value = 0
question_max_len = 40
answer_max_len = 40
embedding_dim = 300
questions = []
answers = []
labels = []

# word_vectors = []
zh_word_vectors = vec.load_word2vec_format(train_file_dir + 'cn.skipgram.bin/cn.skipgram.bin', binary=True,
                                           unicode_errors='ignore')
en_word_vectors = vec.load_word2vec_format(train_file_dir + 'GoogleNews-vectors-negative300.bin', binary=True,
                                           unicode_errors='ignore')


def view_bar(num, total):
    rate = num / total
    rate_num = int(rate * 100)
    r = '\r' + '[' + "=" * (rate_num // 2) + " " * (50 - rate_num // 2) + '] ' + str(rate_num) + '%'
    r = r + ' 已完成：' + str(num) + '/' + str(total)
    sys.stdout.write(r)
    sys.stdout.flush()


def get_word2vec(string):
    try:
        return zh_word_vectors[string]
    except KeyError as e:
        try:
            return en_word_vectors[string]
        except KeyError:
            return [mask_value] * embedding_dim


def delete_no_use_word(string):
    pattern = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。?!@#$%^&*()_+=？、~@#￥%……&*（）]+"
    return re.sub(pattern, "", string)


with open(train_file_dir + 'BoP2017-DBQA.' + kind + '.txt', encoding='utf-8-sig') as train_data_file:
    all_text = [L.rstrip('\n') for L in train_data_file]
    total = len(all_text)
    print(total)
    view_bar_count = 1
    for line in all_text[:5000]:
        # 显示当前完成进度
        view_bar(view_bar_count, total)
        view_bar_count += 1
        # print(view_bar_count)

        triple = line.split('\t')
        label = int(triple[0])
        labels.append(label)

        # 处理问题，从word2vec中获取300维度的向量
        q_list = jieba.lcut(delete_no_use_word(triple[1]))
        question_temp = []
        for q in q_list:
            question_temp.append(get_word2vec(q))
        # 如果，超过指定长度，则截取，不超过，则补 300维全0
        if len(question_temp) > question_max_len:
            question_temp = question_temp[:question_max_len]
        else:
            for i in range(question_max_len - len(question_temp)):
                question_temp.append([mask_value] * embedding_dim)
        if len(question_temp) != question_max_len:
            print('question_temp 出 bug 了')
        questions.append(question_temp)
        # print(question_temp)

        # 处理答案，从word2vec中获取300维度的向量
        q_list = jieba.lcut(delete_no_use_word(triple[1]))
        answer_temp = []
        for q in q_list:
            answer_temp.append(get_word2vec(q))
        # 如果，超过指定长度，则截取，不超过，则补 300维全0
        if len(answer_temp) > answer_max_len:
            answer_temp = answer_temp[:answer_max_len]
        else:
            for i in range(answer_max_len - len(answer_temp)):
                answer_temp.append([mask_value] * embedding_dim)
        if len(answer_temp) != answer_max_len:
            print('answer_temp 出 bug 了')
        answers.append(answer_temp)
        # print(np.shape(answer_temp))

print('开始转为numpy格式')
# 正确的标签集合转化为noe-hot形式
# labels = np_utils.to_categorical(labels)

# print(y)
questions = np.array(questions)
answers = np.array(answers)
labels = np.array(labels)

print(np.shape(questions))
print(np.shape(answers))
print(np.shape(labels))

# data = (questions, answers, labels)
print("开始将所有数据序列化到本地...")
np.save(kind + '_' + 'questions' + '.npy', questions)
np.save(kind + '_' + 'answers' + '.npy', answers)
np.save(kind + '_' + 'labels' + '.npy', labels)
print("序列化到本地成功！...")