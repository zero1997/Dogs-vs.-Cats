from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py


def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    
    height = image_size[1]
    input_tensor = Input((height, width, 3))  # 输入层参数，长宽3通道
    x = input_tensor
    # 匹配输入层，因为后面两个网络的输入层都将数据限定在(-1，1)中
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x,
                       weights='imagenet',
                       include_top=False)
    model = Model(base_model.input,
                  GlobalAveragePooling2D()(base_model.output))
    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("train2",
                                              image_size,
                                              shuffle=False,
                                              batch_size=16)
    test_generator = gen.flow_from_directory("test2",
                                             image_size,
                                             shuffle=False,
                                             batch_size=16,
                                             class_mode=None)

    train = model.predict_generator(train_generator,
                                    train_generator.nb_sample)
    test = model.predict_generator(test_generator,
                                   test_generator.nb_sample)
    
    # 导出模型，都有对应的名字
    with h5py.File("gap_%s.h5"%MODEL.func_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)
    

print(0)
write_gap(ResNet50, (224, 224))
print(1)
write_gap(InceptionV3, (229, 229), inception_v3.preprocess_input)
write_gap(Xception, (299, 299), xception.preprocess_input)
