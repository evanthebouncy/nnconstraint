import tensorflow as tf
import numpy as np
from data import *

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, L, L, 1])
y_true = tf.placeholder(tf.float32, [None, 2])


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

print h_conv1.get_shape()
print h_pool1.get_shape()

W_conv2 = weight_variable([5, 5, 32, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([5 * 5 * 32, 512])
b_fc1 = bias_variable([512])

h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

print y_conv.get_shape()

# Minimize the mean squared errors.
# loss = tf.reduce_mean(tf.square(y - y_data))
# loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_conv), reduction_indices=[1]))
loss = -tf.reduce_sum(y_true * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))


optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

for i in range(5000001):
  # get data dynamically from my data generator
  dat, lab = gen_data()
  sess.run(train, feed_dict={x: dat, y_true: lab, keep_prob: 0.5})
  print sess.run(loss, feed_dict={x: dat, y_true: lab, keep_prob: 1.0})
  
  # do evaluation every 100 epochs
  if (i % 100 == 0):
    print("====current accuracy==== at epoch ", i)
    print "gold standard"
    print lab
    output_y = sess.run([y_conv], feed_dict={x:dat, keep_prob:1.0})
    print(output_y)

  # continuously train at every epoch

