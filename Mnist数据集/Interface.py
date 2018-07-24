# 对函数进行二次的封装

import tensorflow as tf
# 正则化速率 ， type 为l1 或者 l2


def regularizer(regularization_rate, type):
    if type == 'l1':
        return tf.contrib.layers.l1_regularizer(regularization_rate)
    elif type == 'l2':
        return tf.contrib.layers.l2_regularizer(regularization_rate)


with tf.Session() as sess:
    constant = tf.constant([1.0,2.0],dtype = tf.float32)
    print(sess.run(regularizer(0.1,'l2')(constant)))
