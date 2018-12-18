import h5py
import numpy as np 
from sklearn.utils import shuffle
np.random.seed(2017)

#这些导出的特征向量记录了所有图片的内容
X_train = []
y_test = []

for filename in [""]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)