import re
import jieba
import numpy as np

q_len = []
a_len = []

temp = "asdf!@#!!#@#!SFsafg111ferwe"
pattern = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。?!@#$%^&*()_+=？、~@#￥%……&*（）]+"
print()

with open('../../data/BoP2017-DBQA.train.txt', encoding='utf-8-sig') as train_data_file:
    all_text = [L.rstrip('\n') for L in train_data_file]
    # print('总长度', len(all_text))
    temp = ''
    ans_list = []
    for i, line in enumerate(all_text[:]):
        print(i, line)
