# 安装过的包

换源：-i https://pypi.tuna.tsinghua.edu.cn/simple

- keras
- tensorflow（ tensorflow-gpu）
- numpy
- scipy
- h5py
- matplotlib
- scikit-learn 
- gensim
- jieba
- pyemd
- nltk
- Python-Levenshtein

 
# 统计结果

- question 词个数最大：31
- answer 词个数最大：2339 ，平均：17.8640697452


# dev实验结果
- word2vec 不去停用词 : 0.4325970681651023
- word2vec 去停用词 : 0.47675953244116814
- LSI 不去停用词: 0.4664553297676715
- LSI 去停用词: 0.4664553297676715
- tf-idf 不去停用词 : 0.5835605596118271
- tf-idf 去停用词 : 0.5930308209408673
- Levenshtein  : 0.37447474590123164
