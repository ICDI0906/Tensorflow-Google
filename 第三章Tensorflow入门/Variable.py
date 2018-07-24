import tensorflow as tf
w1 = tf.Variable(tf.random_normal((2, 3), stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev = 1, seed = 1))
# x = tf.constant([[2, 3]], dtype = tf.float32) # 常量
x = tf.placeholder(tf.float32, shape = (2, 2), name = "input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
with tf.Session() as sess:
    # tf.initialize_all_variables(); 不能使用该语句进行初始化
    init_op = tf.global_variables_initializer() # 可以使用该方式进行变量的初始化
    sess.run(init_op)
    # sess.run(w1.initializer);sess.run(w2.initializer) # 一个一个地进行变量的初始化
    # print(sess.run(a))
    # print(tf.global_variables()) # 得到所有的参数
    # print(tf.trainable_variables()) # 得到所有的优化参数
    # print(sess.run(y))
    print(sess.run(y, feed_dict = {x: [[1.0, 2.0], [3.5, 4.5]]}))

