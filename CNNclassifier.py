import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
import os
import sys

import keras
import h5py
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, MaxPool2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16,Xception
from sklearn.model_selection import train_test_split
from sklearn import metrics


from keras.optimizers import SGD
from keras.models import load_model
from keras.preprocessing import image
# %%
## 读取数据集的基本信息
# this program return a model file
class CNNclassifier():
    def __init__(self):
        self.monkey_labels = pd.read_csv("monkey_labels.txt")

## 设置图像尺寸为150*150*3和batch_size=32
        self.height = 150
        self.width = 150
        self.channels = 3
        self.batch_size = 32
        self.seed = 1337
        self.num_classes = 3
        self.epochs = 10
        self.train_samplesize = 11851
        self.val_samplesize = 1216
        self.train_path = "./training/"
        self.val_path = "./val/"
        self.model
        self.model_fit

## 使用图片生成器ImageDataGenerator来准备训练集图像数据集
        self.datagen = ImageDataGenerator(
            rotation_range=40,  ## 图片随机转动的角度
            rescale=1. / 255,  ## 图像像素转化到0～1之间
            shear_range=0.2,  ## 剪切强度
            zoom_range=0.2,  ## 随机缩放的幅度
            horizontal_flip=True,  ## 随机水平翻转
            vertical_flip=True,  ## 进行随机竖直翻转
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='nearest'
        )

## 从指定文件路径中读取训练数据集
        self.train_data = self.datagen.flow_from_directory(
            directory=self.train_path,  ## 数据文件路径
            target_size=(self.height, self.width),  ## 图像将被resize成该尺寸
            batch_size=self.batch_size,  ##  batch数据的大小
            seed=self.seed,  ## 打乱数据和进行变换时的随机数种子
            shuffle=True,
            class_mode="categorical"  ## 该参数决定了返回的标签数组的形式,"categorical"会返回2D的one-hot编码标签,
        )

# 使用图片生成器ImageDataGenerator来准备测试集集图像
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)  ## 只将图像像素值转化到0～1之间
        self.test_data = self.test_datagen.flow_from_directory(
            directory=self.val_path,  ## 数据文件路径
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            seed=self.seed,
            class_mode="categorical"
        )


    def extract_features(self,sample_count, datagen, labelclass):
        """
        sample_count:需要生成的样本数量
        datagen:使用图片生成器ImageDataGenerator定义的数据集生成器
        labelclass:数据集的类别数目
        """
        start = time()
        labels = np.zeros(shape=(sample_count, labelclass))
        generator = datagen
        batch_size = generator.batch_size
        i = 0
        for inputs_batch, labels_batch in generator:
            stop = time()

            labels[i * batch_size: (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= sample_count:
                break

        print("\n")
        self.labels =  labels

    def train(self):
        self.train_num = self.train_data.samples
        self.validation_num = self.test_data.samples

        self.test_labels = self.extract_features(self.val_samplesize, self.test_data, 3)

        # 定义输入
        input_shape = (150, 150, 3)  # RGB影像150*150（height,width,channel)

        # 使用序贯模型(sequential)来定义
        model = Sequential(name='vgg16-sequential')

        # 第1个卷积区块(block1)
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape, name='block1_conv1'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2'))
        model.add(MaxPool2D((2, 2), strides=(2, 2), name='block1_pool'))

        # 第2个卷积区块(block2)
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2'))
        model.add(MaxPool2D((2, 2), strides=(2, 2), name='block2_pool'))

        # 第3个区块(block3)
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3'))
        model.add(MaxPool2D((2, 2), strides=(2, 2), name='block3_pool'))

        # 第4个区块(block4)
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3'))
        model.add(MaxPool2D((2, 2), strides=(2, 2), name='block4_pool'))

        # 第5个区块(block5)
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3'))
        model.add(MaxPool2D((2, 2), strides=(2, 2), name='block5_pool'))

        # 前馈全连接区块
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dense(1000, activation='softmax', name='fc3'))
        model.add(Dense(self.num_classes, activation='softmax', name='predictions'))
        model.add(Activation('softmax'))
        # 打印网络结构
        model.summary()
        self.model = model

    ## 对模型进行编译和训练
        self.model.compile(SGD(lr=0.01),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
    # model.compile(optimizer='adam',
    #              loss='categorical_crossentropy',
    #              metrics=['acc'])
    # 通过fit_generator来对数据集进行训练

        self.model_fit = model.fit(self.train_features, self.train_labels,batch_size=32,
                       shuffle=True,validation_split=0.1,  ## 使用百分之10作为验证集
                       epochs=self.epochs, verbose=1)

        return model

    def test(self):

        model = load_model('my_modle.h5')  # 读取已有模型

        self.model_fit = model.fit_generator(
                                        self.train_data,
                                        steps_per_epoch=self.train_num // self.batch_size,
                                        epochs=self.epochs,
                                        validation_data=self.train_data,
                                        validation_steps=self.validation_num // self.batch_size,
                                        verbose=1)

    def save(self):

        # 保存模型
        model_name = time()
        self.model.save('my_modle.h5')

    def show(self):
## 将模型的结果可视化
        model_accdf = pd.DataFrame(self.model_fit.history)
        model_accdf["epochs"] = self.model_fit.epoch
        model_accdf.head()



# %%
## 可视化acc 和loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(model_accdf.epochs, model_accdf.acc, "r-", label="train acc")
        plt.plot(model_accdf.epochs, model_accdf.val_acc, "b--", label="val acc")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("acc")
        plt.subplot(1, 2, 2)

        plt.plot(model_accdf.epochs, model_accdf.loss, "r-", label="train loss")
        plt.plot(model_accdf.epochs, model_accdf.val_loss, "b--", label="val loss")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
    # plt.show()

    ## 对测试集进行预测
        prey = self.model.predict(self.test_data)
        # 从 one-hot 编码中获得类别label
        y_pre = [i.argmax() for i in prey]
        y_true = [i.argmax() for i in self.test_labels]

    ## 计算预测结果混淆矩阵并可视化
    ## 混淆矩阵

        conmat = metrics.confusion_matrix(y_true, y_pre)
        plt.figure(figsize=(8, 8))
        sns.heatmap(conmat.T, square=True, annot=True,
                    fmt='d', cbar=False, linewidths=.5,
                    cmap="YlGnBu")
        plt.xlabel('True label', size=14)
        plt.ylabel('Predicted label', size=14)
        plt.title("Confusion matrix", size=16)
        plt.show()

        files = os.listdir(self.image_path)
        for file in files:  # 遍历文件夹
            #    if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开

            img = image.load_img(self.image_path + "/" + file, target_size=(200, 200))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            feature = self.model.predict(x)
            feature = feature.reshape((6, 6, 512))
            feature = np.expand_dims(feature, axis=0)
            prey = self.model.predict_classes(feature)
            # 从 one-hot 编码中获得类别label
            print(prey, end='')
        mode = "end"



