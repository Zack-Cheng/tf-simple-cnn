#!./tensorflow/venv/bin/python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from lib.Utils import load_train_dataset, to_hot_vector

## Parameters ##
LEARN_RATE = 0.0001
BATCH_SIZE = 100
TRAIN_EPOCH = 20000
DEV_TRAIN_SET_SIZE = 50000
################

################
CONV1_W  = 5
CONV2_W  = 5
POOL1_W = 2
POOL2_W = 2
CONV1_OC = 5
CONV2_OC = 2
CONV1_STRIDE = 1
CONV2_STRIDE = 1
POOL1_STRIDE = 2
POOL2_STRIDE = 1
CONV1_PAD = 'VALID'
CONV2_PAD = 'VALID'
POOL1_PAD = 'VALID'
POOL2_PAD = 'VALID'
FC_HIDDEN_UNIT = 50
CLASS_NUM = 10
################

img_w = 28
img_ch = 1
img = tf.placeholder(dtype='float32', shape=[None, img_w, img_w, img_ch])
t = tf.placeholder(dtype='float32', shape=[None, CLASS_NUM])

W1 = tf.get_variable(
    'W1',
    shape=[CONV1_W, CONV1_W, img_ch, CONV1_OC],
    initializer=tf.contrib.layers.xavier_initializer()
)
W2 = tf.get_variable(
    'W2',
    shape=[CONV2_W, CONV2_W, CONV1_OC, CONV2_OC],
    initializer=tf.contrib.layers.xavier_initializer()
)

conv1_w = (img_w - CONV1_W) // CONV1_STRIDE + 1
pool1_w = (conv1_w - POOL1_W) // POOL1_STRIDE + 1
conv2_w = (pool1_w - CONV2_W) // CONV2_STRIDE + 1
pool2_w = (conv2_w - POOL2_W) // POOL2_STRIDE + 1

W3 = tf.get_variable(
    'W3',
    shape=[(pool2_w ** 2) * CONV2_OC, FC_HIDDEN_UNIT],
    initializer=tf.contrib.layers.xavier_initializer()
)
W4 = tf.get_variable(
    'W4',
    shape=[FC_HIDDEN_UNIT, CLASS_NUM],
    initializer=tf.contrib.layers.xavier_initializer()
)

b1 = tf.Variable(tf.zeros([CONV1_OC]))
b2 = tf.Variable(tf.zeros([CONV2_OC]))
b3 = tf.Variable(tf.zeros([FC_HIDDEN_UNIT]))
b4 = tf.Variable(tf.zeros([CLASS_NUM]))

h_conv1 = tf.nn.relu(
    tf.nn.conv2d(img,
                 W1,
                 strides=[1, CONV1_STRIDE, CONV1_STRIDE, 1],
                 padding=CONV1_PAD) + b1
)
h_pool1 = tf.nn.max_pool(h_conv1,
                     ksize=[1, POOL1_W, POOL1_W, 1],
                     strides=[1, POOL1_STRIDE, POOL1_STRIDE, 1],
                     padding=POOL1_PAD)
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(h_pool1,
                 W2,
                 strides=[1, CONV2_STRIDE, CONV2_STRIDE, 1],
                 padding=CONV2_PAD) + b2
)
h_pool2 = tf.nn.max_pool(h_conv2,
                     ksize=[1, POOL2_W, POOL2_W, 1],
                     strides=[1, POOL2_STRIDE, POOL2_STRIDE, 1],
                     padding=POOL2_PAD)

x_flat_1 = tf.reshape(h_pool2, [-1, (pool2_w ** 2) * CONV2_OC])
h_flat_1 = tf.nn.relu(
    tf.matmul(x_flat_1, W3) + b3
)
h_flat_2 = tf.matmul(h_flat_1, W4) + b4

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=h_flat_2))
train = tf.train.AdamOptimizer(LEARN_RATE).minimize(cross_entropy)
predict = tf.argmax(h_flat_2, 1)
correct_predict = tf.equal(predict, tf.argmax(t, 1))
acc = tf.reduce_mean(tf.cast(correct_predict, 'float32'))

init = tf.global_variables_initializer()

dataset, label = load_train_dataset()
acc_list = []
epoch_list = []

with tf.Session() as sess:
    sess.run(init)

    for i in range(TRAIN_EPOCH):
        indices = np.random.choice(DEV_TRAIN_SET_SIZE, BATCH_SIZE)
        batch_dataset = dataset[indices]
        batch_label = to_hot_vector(label[indices])
        if i % 100 == 0:
            accuracy = sess.run(acc, feed_dict={
                img: batch_dataset,
                t: batch_label
            })
            acc_list.append(accuracy)
            epoch_list.append(i)
            print(
                'epoch: {}, dev set accuracy: {}'.format(
                    i,
                    accuracy
                )
            )
        sess.run(train, feed_dict={
            img: batch_dataset,
            t: batch_label
        })

    print(
        'test set accuracy: {}'.format(
            sess.run(acc, feed_dict={
                img: dataset[DEV_TRAIN_SET_SIZE:],
                t: to_hot_vector(label[DEV_TRAIN_SET_SIZE:])
            })
        )
    )

    plt.plot(epoch_list, acc_list)
    plt.show()
