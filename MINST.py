import os
import random
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

image_size_input = 28
batch_size_input = 64
num_epoch_input = 10
num_class_input = 10
channel_input = 1
mode_input = 'train'  # or test
learning_rate_input = 0.0001
if_test_input = 0
data_path_input = 'data\\number\\Testimage'


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


def read_batch(img_list, labels, num_class, batch_size, num_epoch):
    data_size = len(img_list)
    for epoch in range(num_epoch):
        data = list(zip(img_list, labels))
        random.shuffle(data)
        img_list[:], labels[:] = zip(*data)
        print(epoch, "epoch is completed")
        for batch in range(0, data_size - batch_size, batch_size):
            imgs = []
            for i in range(batch_size):
                img = cv2.imread(img_list[batch + i], 0)
                img = np.expand_dims(img, -1)
                imgs.append(img)
            img_batch = np.array(imgs, dtype=np.float32) / 255.0
            label_batch = np.array(labels[batch: batch + batch_size])
            label_batch = (np.arange(num_class) == label_batch[:, None]).astype(np.float32)
            yield img_batch, label_batch


def read_test(img_list, labels, num_class, batch_size):
    data_size = len(img_list)
    data = list(zip(img_list, labels))
    random.shuffle(data)
    img_list[:], labels[:] = zip(*data)
    for batch in range(0, data_size, batch_size):
        if batch + batch_size > data_size:
            img_batch = img_list[batch: data_size]
            label_batch = np.array(labels[batch: data_size])
        else:
            img_batch = img_list[batch: batch + batch_size]
            label_batch = np.array(labels[batch: batch + batch_size])
        imgs = []
        for img_name in img_batch:
            img = cv2.imread(img_name, 0)
            img = np.expand_dims(img, -1)
            imgs.append(img)
        img_batch = np.array(imgs, dtype=np.float32) / 255.0
        label_batch = (np.arange(num_class) == label_batch[:, None]).astype(np.float32)
        yield img_batch, label_batch


def conv_backup(input, out_channel, name="conv"):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv2d(input, out_channel, kernel_size=5, strides=(1, 1), padding='SAME',
                                activation=tf.nn.relu, kernel_initializer=initializer)


def linear_backup(input, out_size, activation, name='linear'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.dense(input, out_size, activation=activation, kernel_initializer=initializer, name='out')


def conv(raw_input, output_channel, name="conv"):
    with tf.variable_scope(name):
        input_channel = raw_input.get_shape().as_list()[-1]

        weight = tf.get_variable('weight', [5, 5, input_channel, output_channel],
                                 initializer=tf.random_normal_initializer(0, 0.02))

        conv_func = tf.nn.conv2d(raw_input, weight, [1, 1, 1, 1], padding='SAME')

        bias = tf.get_variable('bias', [output_channel], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv_func, bias)

        return tf.nn.relu(out)


def linear(linear_input, out_size, activation=None, name='linear'):
    with tf.variable_scope(name):
        input_size = linear_input.get_shape().as_list()[-1]
        weight = tf.get_variable('weight', [input_size, out_size], initializer=tf.random_normal_initializer(0, 0.02))
        bias = tf.get_variable('bias', [out_size], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(tf.matmul(linear_input, weight), bias)
        return activation(out)


def pool(pool_input, name="pool"):
    return tf.nn.max_pool(pool_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def model(raw_data, target, name="model"):
    with tf.variable_scope(name):  # assigned the layer with its name "model"

        out_1 = conv(raw_data, 32, 'out_1')
        pool_1 = pool(out_1, 'pool_1')

        out_2 = conv(pool_1, 64, 'out_2')
        pool_2 = pool(out_2, 'pool_2')

        flat = tf.layers.flatten(pool_2, 'flat')

        fc_1 = linear(flat, 1024, activation=tf.nn.relu, name='fc_1')

        if not if_test_input:
            drop = tf.nn.dropout(fc_1, 0.5, name='drop')
        else:
            drop = fc_1

        output = linear(drop, num_class_input, activation=tf.nn.softmax, name='output')
        loss = tf.reduce_sum(-target * tf.log(output))
        prediction = tf.equal(tf.arg_max(output, 1), tf.arg_max(target, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_input).minimize(loss)
        return optimizer, loss, accuracy


def main():
    start_time = time.time()
    img_list, label_list = read_img(data_path_input)

    save_path = os.path.join(sys.path[0], "save")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, "model.ckpt")

    image = tf.placeholder(tf.float32, [None, image_size_input, image_size_input, channel_input], name='input')
    label = tf.placeholder(tf.float32, [None, num_class_input], name='label')

    optimizer, loss, accuracy = model(image, label)

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    if mode_input == "train":
        sess.run(tf.global_variables_initializer())
        step = 0
        for batch_input, batch_label in read_batch(img_list, label_list, num_class_input, batch_size_input,
                                                   num_epoch_input):
            step += 1
            sess.run(optimizer, feed_dict={image: batch_input, label: batch_label})
            if step % 100 == 0:
                accu = sess.run(accuracy, feed_dict={image: batch_input, label: batch_label})
                print("step: ", step, ", accuracy: ", accu, ", time: ", time.time() - start_time)
            if step % 2000 == 0:
                saver.save(sess, save_path)

        saver.save(sess, save_path)
        print("Training complete, time ", time.time() - start_time, "s")

    if mode_input == "test":
        if_test = 1
        saver.restore(sess, save_path)
        accu = 0
        for img_batch, label_batch in read_test(img_list, label_list, num_class_input, 100):
            accu += img_batch.shape[0] * sess.run(accuracy, feed_dict={image: img_batch, label: label_batch})
        print("The test accuracy is ", accu / len(img_list))


if __name__ == '__main__':
    main()
