import tensorflow as tf
import numpy as np
from data import *
from draw import *

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# the shape and strides must match up
def conv2d_trans(x, W, shape, strides):
  return tf.nn.conv2d_transpose(x, W, shape, strides, padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')


x = tf.placeholder(tf.float32, [50, L, L, 1])

print "x_shape ", x.get_shape()
W_conv1 = weight_variable([5, 5, 1, 2])
b_conv1 = bias_variable([2])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

print "hidden shape ", h_pool1.get_shape()

# notice it's 5, 5, 1, 2
# where it is 5, 5, OUTPUT, INPUT
# the input/ouput is flipped the other way
W_deconv1 = weight_variable([5, 5, 2, 2])
b_deconv1 = bias_variable([2])
regen1 = tf.nn.relu(conv2d_trans(h_pool1, W_deconv1, [50,L/2,L/2,2], [1,2,2,1]) + b_deconv1)

print "regen1 shape ", regen1.get_shape()

W_deconv2 = weight_variable([5, 5, 1, 2])
b_deconv2 = bias_variable([1])
regen = tf.nn.relu(conv2d_trans(regen1, W_deconv2, [50,L,L,1], [1,2,2,1]) + b_deconv2)

print "output shape ", regen.get_shape()

assert regen.get_shape() == x.get_shape()

diff_cost = tf.reduce_mean(tf.square(x - regen), [0,1,2,3])
print "cost shape ", diff_cost.get_shape()

optimizer = tf.train.GradientDescentOptimizer(0.005)
train = optimizer.minimize(diff_cost)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# dat, lab = gen_data()
# print sess.run([loss], feed_dict={x: dat, y_true: lab, keep_prob: 1.0})
# print sess.run([y_conv], feed_dict={x: dat, y_true: lab, keep_prob: 1.0})
# sess.run([train, loss], feed_dict={x: dat, y_true: lab, keep_prob: 0.5})
# print sess.run([loss], feed_dict={x: dat, y_true: lab, keep_prob: 1.0})
# print sess.run([y_conv], feed_dict={x: dat, y_true: lab, keep_prob: 1.0})

for i in range(5000001):
  # get data dynamically from my data generator
  dat, lab = gen_data()
  feed_dict = {x:dat}
  sess.run(train, feed_dict=feed_dict)
  print sess.run(diff_cost, feed_dict=feed_dict)
  
  # do evaluation every 100 epochs
  if (i % 500 == 0):
    img_regen = sess.run(regen, feed_dict=feed_dict)
    draw(dat[0], "./drawings/{0}_orig.png".format(i/500))
    draw(img_regen[0], "./drawings/{0}_regen.png".format(i/500))

