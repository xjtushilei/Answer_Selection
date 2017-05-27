import numpy as np

pre = np.load('predict.npy')
for x in pre[:45]:
    print(x[0])