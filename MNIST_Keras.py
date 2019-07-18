import os
import random

import cv2
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import np_utils

np.random.seed(10)


def read_img(data_path):
    sub_list = os.listdir(data_path)
    imgs = []
    labels = []
    num = 0
    for sub_folder in sub_list:
        sub_path = os.path.join(data_path, sub_folder)
        img_list = os.listdir(sub_path)
        for img_name in img_list:
            img_path = os.path.join(sub_path, img_name)
            imgs.append(img_path)
            labels.append(num)
        num += 1
    return imgs, labels


def main():
    img_list, label_list = read_img('data\\number\\Testimage')
    data = list(zip(img_list, label_list))
    random.shuffle(data)
    x_train = []
    y_train = []
    for img_path, label in data:
        im = cv2.imread(img_path, 0)
        x_train.append(im)
        y_train.append(label)

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    x_train_4d = x_train.reshape((x_train.shape[0], 28, 28, 1))

    x_train_normlize = x_train_4d / 255.0
    y_train_onehot = np_utils.to_categorical(y_train)

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary(), '\n')

    model.fit(x=x_train_normlize, y=y_train_onehot, validation_split=0.2, epochs=10, batch_size=64,
              verbose=2)

    img_list, label_list = read_img('data\\number\\Trainimage')
    data = list(zip(img_list, label_list))
    random.shuffle(data)
    x_test = []
    y_test = []
    for img_path, label in data:
        im = cv2.imread(img_path, 0)
        x_test.append(im)
        y_test.append(label)

    y_test = np.array(y_test, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)

    x_test_4d = x_test.reshape((x_test.shape[0], 28, 28, 1))

    x_test_normlize = x_test_4d / 255.0

    prediction = model.predict_classes(x_test_normlize)

    print('prediction: {} vs actual: {}'.format(prediction[:10], y_test[:10]))

    print(pd.crosstab(y_test, prediction, rownames=['label'], colnames=['predict']))


if __name__ == '__main__':
    main()
