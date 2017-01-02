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


# ---------------------------------------------------------- list of vars to keep track
train_gen_vars = []
train_dis_vars = []


# ------------------------------------------------------------------ input
x = tf.placeholder(tf.float32, [50, L, L, 1])

# ------------------------------------------------------------------ generative network

print "x_shape ", x.get_shape()
W_conv1 = weight_variable([5, 5, 1, 2])
b_conv1 = bias_variable([2])
train_gen_vars += [W_conv1, b_conv1]

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

print "hidden shape ", h_pool1.get_shape()

# notice it's 5, 5, 1, 2
# where it is 5, 5, OUTPUT, INPUT
# the input/ouput is flipped the other way
W_deconv1 = weight_variable([5, 5, 2, 2])
b_deconv1 = bias_variable([2])
train_gen_vars += [W_deconv1, b_deconv1]

regen1 = tf.nn.relu(conv2d_trans(h_pool1, W_deconv1, [50,L/2,L/2,2], [1,2,2,1]) + b_deconv1)

print "regen1 shape ", regen1.get_shape()

W_deconv2 = weight_variable([5, 5, 1, 2])
b_deconv2 = bias_variable([1])
train_gen_vars += [W_deconv2, b_deconv2]

regen = tf.nn.relu(conv2d_trans(regen1, W_deconv2, [50,L,L,1], [1,2,2,1]) + b_deconv2)

# genertive network error
print "output shape ", regen.get_shape()

assert regen.get_shape() == x.get_shape()

diff_cost = tf.reduce_mean(tf.square(x - regen), [0,1,2,3])
print "cost shape ", diff_cost.get_shape()

# ---------------------------------------------------------------- disciminant network

# ----------------------------------------- predict true on original Xs
W_conv_dis1 = weight_variable([5,5,1,2])
b_conv_dis1 = bias_variable([2])
train_dis_vars += [W_conv_dis1, b_conv_dis1]

x_dis_conv1 = tf.nn.relu(conv2d(x, W_conv_dis1) + b_conv_dis1)
x_dis_pool1 = max_pool_2x2(x_dis_conv1)

print "x_dis_pool1 shape ", x_dis_pool1.get_shape()

x_dis_flat = tf.reshape(x_dis_pool1, [50,5*5*2])

W_fc_dis = weight_variable([5*5*2, 2])
b_fc_dis = bias_variable([2])
train_dis_vars += [W_fc_dis, b_fc_dis]

x_dis_pred = tf.nn.softmax(tf.matmul(x_dis_flat, W_fc_dis) + b_fc_dis)

print "x_dis_pred shape ", x_dis_pred.get_shape()

# demand x_dis_pred to be TRUE
true_lab = tf.tile(tf.constant([[1.0, 0.0]]), [50, 1])
print "true_lab shape ", true_lab.get_shape()

cost_dis_true = tf.reduce_mean(tf.square(true_lab - x_dis_pred))

# ------------------------------------------ predict false on the generated Xs
regen_dis_conv1 = tf.nn.relu(conv2d(regen, W_conv_dis1) + b_conv_dis1)
regen_dis_pool1 = max_pool_2x2(regen_dis_conv1)

print "regen_dis_pool1 shape ", regen_dis_pool1.get_shape()

regen_dis_flat = tf.reshape(regen_dis_pool1, [50,5*5*2])
regen_dis_pred = tf.nn.softmax(tf.matmul(regen_dis_flat, W_fc_dis) + b_fc_dis)

print "regen_dis_pred shape ", regen_dis_pred.get_shape()

# demand regen_dis_pred to be FALSE
false_lab = tf.tile(tf.constant([[0.0, 1.0]]), [50, 1])
print "false_lab shape ", false_lab.get_shape()

cost_dis_false = tf.reduce_mean(tf.square(false_lab - regen_dis_pred))

# total dis cost
cost_dis = cost_dis_true + cost_dis_false

# ------------------------------------------------------------ fooling cost
# i.e. train generater to have regen_dis_pred to be TRUE instead of FALSE
fool_cost = tf.reduce_mean(tf.square(true_lab - regen_dis_pred))
cost_gen = fool_cost + diff_cost

# ------------------------------------------------------------ training

optimizer = tf.train.GradientDescentOptimizer(0.005)
train_gen_gen = optimizer.minimize(diff_cost, var_list = train_gen_vars)
train_gen_fool = optimizer.minimize(fool_cost, var_list = train_gen_vars)
train_dis = optimizer.minimize(cost_dis, var_list = train_dis_vars)

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

  print "training generator gen ",
  sess.run([train_gen_gen], feed_dict=feed_dict)
  print sess.run([diff_cost], feed_dict=feed_dict)

  print "training generator fool ",
  sess.run([train_gen_fool], feed_dict=feed_dict)
  print sess.run([fool_cost], feed_dict=feed_dict)

  print "training discriminator ",
  sess.run([train_dis], feed_dict=feed_dict)
  print sess.run([cost_dis], feed_dict=feed_dict)
  
  # do evaluation every 100 epochs
  if (i % 500 == 0):
    img_regen = sess.run(regen, feed_dict=feed_dict)
    draw(dat[0], "./drawings/{0}_orig.png".format(i/500))
    draw(img_regen[0], "./drawings/{0}_regen.png".format(i/500))

