import tensorflow as tf
v1 = tf.constant([1.0, 4.0, 3.0, 4.0])
v2 = tf.constant([2.0, 3.0, 4.0, 5.0])
weight = tf.constant([[1.0,-2.0], [-3.0, 4.0]])
sess = tf.InteractiveSession()
print(tf.greater(v1, v2).eval())   # 两个向量逐个比较大小
print(tf.where(tf.greater(v1, v2), v1, v2).eval()) # condition tensor1 tensor2
sess.close()
