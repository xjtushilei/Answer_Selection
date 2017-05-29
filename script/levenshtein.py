import sys

import Levenshtein
from gensim.models.keyedvectors import KeyedVectors
import jieba

import mrr_utils

stopwords = {}.fromkeys([line.rstrip() for line in open('../data/stopword.txt')])


def view_bar(num, total):
    rate = num / total
    rate_num = int(rate * 100)
    r = '\r' + '[' + "=" * (rate_num // 2) + " " * (50 - rate_num // 2) + '] ' + str(rate_num) + '%'
    r = r + ' 已完成：' + str(num) + '/' + str(total)
    sys.stdout.write(r)
    sys.stdout.flush()


def cut_with_stop_words(string):
    segs = jieba.lcut(string)
    final = ''
    for seg in segs:
        if seg not in stopwords:
            final = final + seg
    return final


word_vectors = KeyedVectors.load_word2vec_format('../data/cn.skipgram.bin/cn.skipgram.bin', binary=True,
                                                 unicode_errors='ignore')

deal_file = '../data/BoP2017-DBQA.dev.txt'

with open(deal_file, encoding='utf-8-sig') as train_data_file:
    all_text = [L.rstrip('\n') for L in train_data_file]
    print('deal_file总长度', len(all_text))
    predict_list = []
    for i, line in enumerate(all_text[:]):
        view_bar(i, len(all_text))
        triple = line.split('\t')
        label = int(triple[0])
        q = triple[1]
        a = triple[2]
        fenci_q = cut_with_stop_words(q)
        fenci_a = cut_with_stop_words(a)
        xiangsidu = Levenshtein.jaro_winkler(fenci_q, fenci_a)
        print(xiangsidu)
        predict_list.append(xiangsidu)

with open('levenshtein.txt', 'w') as result_file:
    for x in predict_list:
        result_file.write(str(x) + '\n')
# print(mrr_utils.get_mrr_pre_from_list(predict_list, true_file='../data/BoP2017-DBQA.dev.txt'))
