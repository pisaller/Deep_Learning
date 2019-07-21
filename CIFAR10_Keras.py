import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import np_utils

np.random.seed(10)


def show_predicted_probability(y, prediction, x_img, predicted_probaility, i):
    print('label:', label_dict[y[i]], 'predict', label_dict[prediction[i]])
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(x_img[i], (32, 32, 3)))
    plt.show()
    for j in range(10):
        print(label_dict[j] + ' Probaility:{}%'.format(round(100 * predicted_probaility[i][j], 2)))


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_images_labels_prediction(images, labels, prediction, idx):
    plt.figure(figsize=(16, 9))

    for i in range(0, 10):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[idx], cmap='binary')
        title = str(i) + ',' + label_dict[labels[i]]

        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[i]]
        plt.title(title, fontsize=10)
        idx += 1

    plt.show()


def read_img(data_path):
    sub_list = os.listdir(data_path)
    imgs_list = []
    labels_list = []
    num = 0
    for sub_folder in sub_list:
        sub_path = os.path.join(data_path, sub_folder)
        img_list = os.listdir(sub_path)
        for img_name in img_list:
            img_path = os.path.join(sub_path, img_name)
            imgs_list.append(img_path)
            labels_list.append(num)
        num += 1
    return imgs_list, labels_list


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.25))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary(), '\n')

    return model


def main():
    print('Now Training...')
    img_list, label_list = read_img('..\\Deep_Learning_data\\object\\test')
    data = list(zip(img_list, label_list))
    random.shuffle(data)
    x_train = []
    y_train = []

    for img_path, label in data:
        im = cv2.imread(img_path, 3)
        x_train.append(im)
        y_train.append(label)

    global label_dict
    label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                  8: 'ship', 9: 'truck'}

    plot_images_labels_prediction(x_train, y_train, [], 0)

    # convert list to numpy array
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    x_train_normlize = x_train / 255.0
    y_train_onehot = np_utils.to_categorical(y_train)

    cnn_model = build_model()

    train_history = cnn_model.fit(x=x_train_normlize, y=y_train_onehot, validation_split=0.2, epochs=20, batch_size=128,
                                  verbose=2)

    show_train_history(train_history, 'acc', 'val_acc')

    print('Now Predicting...')
    img_list, label_list = read_img('..\\Deep_Learning_data\\object\\train')
    data = list(zip(img_list, label_list))
    random.shuffle(data)
    x_test = []
    y_test = []
    for img_path, label in data:
        im = cv2.imread(img_path, 3)
        x_test.append(im)
        y_test.append(label)

    y_test = np.array(y_test, dtype=np.float32)

    y_test_onehot = np_utils.to_categorical(y_test)

    x_test = np.array(x_test, dtype=np.float32)

    x_test_normlize = x_test / 255.0

    score = cnn_model.evaluate(x_test_normlize, y_test_onehot, verbose=0)

    print('Model Score: {}'.format(score))

    prediction = cnn_model.predict_classes(x_test_normlize)

    pred_list = []

    actu_list = []

    for i in prediction[:10]:
        pred_list.append(label_dict[i])

    for i in y_test[:10].astype(int):
        actu_list.append(label_dict[i])

    print('prediction: {} vs actual: {}'.format(pred_list, actu_list))

    print(pd.crosstab(y_test, prediction, rownames=['label'], colnames=['predict']))

    predicted_probability = cnn_model.predict(x_test_normlize)

    show_predicted_probability(y_test, prediction, x_test_normlize, predicted_probability, 0)


if __name__ == '__main__':
    main()
