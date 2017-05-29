import Levenshtein


a = ['中国', '打败', '美国']
b = ['游戏', '好玩']
aa = '中国打败美国'
bb ='游戏好玩吗hen好哇啊啊'
bbb ='游戏好玩吗hen好哇啊啊  '
print(Levenshtein.jaro_winkler(bb, bbb))