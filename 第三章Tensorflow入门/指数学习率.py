import tensorflow as tf
import matplotlib.pyplot as plt
learning_rate = 0.1
global_step = 1000
decay_step = 100
decay_rate = 0.96

global_ = tf.Variable(tf.constant(0))
x = tf.train.exponential_decay(learning_rate,global_,decay_step,decay_rate,staircase = True)
x1 = tf.train.exponential_decay(learning_rate,global_,decay_step,decay_rate,staircase = False)

T_C = []
T_X = []
with tf.Session() as sess:
    for i in range (global_step):
        T_C.append(sess.run(x,feed_dict = {global_: i}))
        T_X.append(sess.run(x1,feed_dict = {global_: i}))
plt.figure(1)
plt.plot(range(global_step),T_C,'r--')
plt.plot(range(global_step),T_X,'b--')
plt.show()