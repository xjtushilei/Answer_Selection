def get_mrr_pre(model='dev'):
    want = []
    with open('../../data/BoP2017-DBQA.' + model + '.txt', encoding='utf-8-sig') as train_data_file:
        all_text = [L.rstrip('\n') for L in train_data_file]
        # print('总长度', len(all_text))
        temp = ''
        temp_list = []
        for i, line in enumerate(all_text[:]):
            triple = line.split('\t')
            label = int(triple[0])
            # print(label)
            q = triple[1]
            if temp != q:
                temp = q

                # 将该问题的长度和正确的答案放进去
                if i != 0:
                    # index = temp_list.index(1)
                    try:
                        index = temp_list.index(1)
                        want.append((len(temp_list), index))
                    except ValueError:
                        want.append((len(temp_list), -1))
                # 开始新的问题的统计
                temp_list.clear()
                temp_list.append(label)
            else:
                temp = q
                temp_list.append(label)
                # print(line)
        index = temp_list.index(1)
        want.append((len(temp_list), index))
    return want

# print(get_mrr_pre())