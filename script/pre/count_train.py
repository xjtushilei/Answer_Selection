import re
import jieba
import numpy as np

q_len = []
a_len = []

temp = "asdf!@#!!#@#!SFsafg111ferwe"
pattern = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。?!@#$%^&*()_+=？、~@#￥%……&*（）]+"
print()


with open('../../data/BoP2017-DBQA.train.txt', encoding='utf-8-sig') as train_data_file:
    # train_data_file = train_data_file.read()
    all_text = [L.rstrip('\n') for L in train_data_file]
    print(len(all_text))
    for line in all_text[:10]:
        triple = line.split('\t')
        label = triple[0]
        print(int(label))
        q = triple[1]
        a = triple[2]
        q = re.sub(pattern, "", q)
        a = re.sub(pattern, "", a)
        q_fenci = jieba.lcut(q)
        a_fenci = jieba.lcut(a)
        q_len.append(len(q_fenci))
        a_len.append(len(a_fenci))
        # q_len.append(len(re.sub(pattern, "", q)))
        # a_len.append(len(re.sub(pattern, "", a)))
print(np.max(q_len))
print(np.max(a_len), np.mean(a_len))
print(len(a_len))
