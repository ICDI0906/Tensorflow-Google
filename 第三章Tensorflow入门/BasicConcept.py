import tensorflow as tf
a = tf.constant([1.0, 2.0], name = "a")
b = tf.constant([3.0, 4.0], name = "b")
result = tf.add(a, b, name = "add")
# print(a.graph is tf.get_default_graph())
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", shape = [1], initializer = tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", shape = [1], initializer = tf.ones_initializer)

with tf.Session(graph = g1) as sess:
    tf.global_variables_initializer().run() # 初始化所有的变量
    with tf.variable_scope("", reuse = True):
        print(sess.run(tf.get_variable("v")))


# sess = tf.Session()
with tf.Session() as sess:
    # sess.run(result)
    print(result.eval())

sess = tf.Session()
with sess.as_default():
    print(result.eval())