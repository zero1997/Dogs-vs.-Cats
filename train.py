'''
import os
import shutil
# 图片数据集预处理
train_filenames = os.listdir("train")
train_cat = filter(lambda x: x[:3] == 'cat', train_filenames)
train_dog = filter(lambda x: x[:3] == 'dog', train_filenames)

def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

rmrf_mkdir("train2")
os.mkdir("train2/cat")
os.mkdir("train2/dog")
rmrf_mkdir('test2')
os.symlink('../test/', 'test2/test')

for filename in train_cat:
    os.symlink('../../train/'+filename, 'train2/cat/'+filename)
for filename in train_dog:
    os.symlink('../../train/'+filename, 'train2/dog/'+filename)
'''
