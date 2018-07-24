import tensorflow as tf
v1 = tf.constant([1.0, 4.0, 3.0, 4.0])
v2 = tf.constant([2.0, 3.0, 4.0, 5.0])
weight = tf.constant([[1.0,-2.0], [-3.0, 4.0]])
sess = tf.InteractiveSession()
print(tf.greater(v1, v2).eval())
print(tf.where(tf.greater(v1, v2), v1, v2).eval())
print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weight)))
# print(sess.run(tf.contrib.layers.l2.regularizer(.5)(weight))) 没有该正则化
sess.close()
