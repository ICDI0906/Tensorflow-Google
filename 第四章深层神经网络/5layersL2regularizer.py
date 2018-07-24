# 损失函数的正则化
import tensorflow as tf

def get_weight(shape, lamb):
    var = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l1.regularizer(lamb)(var))
    return var

x = tf.placeholder(tf.float32, shape = [None, 2])
y_ = tf.placeholder(tf.float32, shape = [None, 1])
batch_size = 8
layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)
cur_layer = x

for layer in range(1, n_layers):

    weight = get_weight([layer_dimension[layer-1], layer_dimension[layer]], 0.01)
    bias = tf.Variable(tf.constant(0.1, shape = [layer_dimension[layer]]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection("losses", mse_loss)

# 将所有的损失函数加起来
loss = tf.add_n(tf.get_collection("losses"))
