import tensorflow as tf
import numpy as np
v1 = tf.placeholder(tf.float32, [None, 2], name = "x")
v2 = tf.placeholder(tf.float32, [None, 2], name = "y")
with tf.Session() as sess:
    tf.global_variables_initializer().run() # 初始化所有的变量
    corrent = tf.equal(v1, v2)
    accracy = tf.reduce_mean(tf.cast(corrent,tf.float32)) # 可以用来计算平均值
    ans = sess.run(accracy, feed_dict = {v1:[[2.0,4.0]], v2:[[2.0,3.0]]})
    print(ans)
