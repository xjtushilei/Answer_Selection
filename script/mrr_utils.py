def get_mrr_pre_from_file(predict_label_file, true_file='../data/BoP2017-DBQA.dev.txt', encoding='utf-8-sig'):
    import numpy
    mrr = []
    with open(true_file, encoding=encoding) as train_data_file, open(predict_label_file) as predict_file:
        all_text = [L.rstrip('\n') for L in train_data_file]
        all_predict_text = [L.rstrip('\n') for L in predict_file]
        # print('总长度', len(all_text))
        temp = ''
        temp_list = []
        predict_list = []
        for i, line in enumerate(all_text[:]):
            triple = line.split('\t')
            label = int(triple[0])
            q = triple[1]
            if temp != q:
                temp = q
                if i != 0:
                    try:
                        index = temp_list.index(1)
                        pridict_true_num = predict_list[index]
                        predict_list.sort(reverse=True)
                        rank_i = predict_list.index(pridict_true_num) + 1
                        # print('总长度：', len(temp_list), '正确排在第n位置：', index + 1, 'rank_i: ', rank_i)
                        mrr.append(1 / rank_i)
                    except ValueError:
                        mrr.append(0)
                # 开始新的问题的统计
                temp_list.clear()
                temp_list.append(label)
                predict_list.clear()
                predict_list.append(float(all_predict_text[i]))
            else:
                temp = q
                temp_list.append(label)
                predict_list.append(float(all_predict_text[i]))
                # print(line)
        try:
            index = temp_list.index(1)
            pridict_true_num = predict_list[index]
            predict_list.sort(reverse=True)
            rank_i = predict_list.index(pridict_true_num) + 1
            # print('总长度：', len(temp_list), '正确排在第n位置：', index, 'rank_i: ', rank_i)
            mrr.append(1 / rank_i)
        except ValueError:
            mrr.append(0)
    print('MRR list是', mrr)
    print('MRR 是', numpy.mean(mrr))
    return numpy.mean(mrr)


# get_mrr_pre_from_file(predict_label_file='pre.txt', true_file='../data/BoP2017-DBQA.dev.txt')


def get_mrr_pre_from_list(predict_label_list, true_file='../data/BoP2017-DBQA.dev.txt', encoding='utf-8-sig'):
    import numpy
    mrr = []
    with open(true_file, encoding=encoding) as train_data_file:
        all_text = [L.rstrip('\n') for L in train_data_file]
        all_predict_text = predict_label_list
        # print('总长度', len(all_text))
        temp = ''
        temp_list = []
        predict_list = []
        for i, line in enumerate(all_text[:]):
            triple = line.split('\t')
            label = int(triple[0])
            q = triple[1]
            if temp != q:
                temp = q
                if i != 0:
                    try:
                        index = temp_list.index(1)
                        pridict_true_num = predict_list[index]
                        predict_list.sort(reverse=True)
                        rank_i = predict_list.index(pridict_true_num) + 1
                        # print('总长度：', len(temp_list), '正确排在第n位置：', index + 1, 'rank_i: ', rank_i)
                        mrr.append(1 / rank_i)
                    except ValueError:
                        mrr.append(0)
                # 开始新的问题的统计
                temp_list.clear()
                temp_list.append(label)
                predict_list.clear()
                predict_list.append(float(all_predict_text[i]))

            else:
                temp = q
                temp_list.append(label)
                predict_list.append(float(all_predict_text[i]))
                # print(line)
        try:
            index = temp_list.index(1)
            pridict_true_num = predict_list[index]
            predict_list.sort(reverse=True)
            rank_i = predict_list.index(pridict_true_num) + 1
            # print('总长度：', len(temp_list), '正确排在第n位置：', index, 'rank_i: ', rank_i)
            mrr.append(1 / rank_i)
        except ValueError:
            mrr.append(0)
    print('MRR list是', mrr)
    print('MRR 是', numpy.mean(mrr))
    return numpy.mean(mrr)
