import jieba
from gensim import corpora, models, similarities

stopwords = {}.fromkeys([line.rstrip() for line in open('../data/stopword.txt')])


def cut_with_stop_words(string):
    segs = jieba.lcut(string)
    final = []
    if True:
        for seg in segs:
            if seg not in stopwords:
                final.append(seg)
        return final
    else:
        return segs


def get_similarity(query, ans_list):
    s_lenth = len(ans_list)
    Corp = ans_list
    # 生成一个有词频的字典
    dictionary = corpora.Dictionary(Corp)
    # 将答案转化成有词频的
    corpus = [dictionary.doc2bow(text) for text in Corp]

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    vec_bow = dictionary.doc2bow(query)
    vec_tfidf = tfidf[vec_bow]

    index = similarities.MatrixSimilarity(corpus_tfidf)
    sims = index[vec_tfidf]
    similarity = list(sims)
    # print(similarity)
    end_lenth = len(similarity)
    if s_lenth != end_lenth:
        print('bug')
    return similarity


deal_file = '../data/BoP2017-DBQA.dev.txt'
result = []
anssumn = 0

with open(deal_file, encoding='utf-8-sig') as train_data_file:
    all_text = [L.rstrip('\n') for L in train_data_file]
    print('总长度', len(all_text))
    temp = ''
    ans_list = []
    for i, line in enumerate(all_text[:]):
        triple = line.split('\t')
        q = triple[1]
        a = triple[2]
        if temp != q:
            temp = q
            if i != 0:
                anssumn += len(ans_list)
                try:
                    fangjinqu = get_similarity(query, ans_list)
                    result.extend(fangjinqu)
                except Exception:
                    # if len(ans_list) != 1:
                    #     print(len(ans_list))
                    #     print(q)
                    #     print(ans_list)
                    #     print('len bug')
                    result.extend([1.0] * len(ans_list))
            # 开始新的问题的统计
            ans_list.clear()
            query = cut_with_stop_words(q)
            ans_list.append(cut_with_stop_words(a))
        else:
            temp = q
            query = cut_with_stop_words(q)
            ans_list.append(cut_with_stop_words(a))
            # print(line)
    anssumn += len(ans_list)
    result.extend(get_similarity(query, ans_list))

print('i:', i)
print('anssumn:', anssumn)
print('result长度：', len(result))

with open('tf-idf.txt', 'w') as result_file:
    for x in result:
        result_file.write(str(x) + '\n')
