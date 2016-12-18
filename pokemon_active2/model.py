import tensorflow as tf
import numpy as np
from data import *

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  # initial = tf.truncated_normal(shape, mean=0.2, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# set up weights for input outputs!
in_obs = tf.placeholder(tf.float32, [n_batch, L, L, 2])
in_query_loc = tf.placeholder(tf.float32, [n_batch, L, L, 1])
out_query_TF = tf.placeholder(tf.float32, [n_batch, 2])
out_z_loc = tf.placeholder(tf.float32, [n_batch, L, L, 1])
keep_prob = tf.placeholder(tf.float32)

def gen_feed_dict(obs, query_loc, query_TF, z_loc):
  ret = {}
  ret[in_obs] = obs
  ret[in_query_loc] = query_loc
  ret[out_query_TF] = query_TF
  ret[out_z_loc] = z_loc
  ret[keep_prob] = 0.7
  return ret

# --------------------------------------------------------------------- generate observation embedding
# some prior distribution
Z_hidden = tf.Variable(tf.truncated_normal([1, L, L, 4], stddev=0.1))
Z_hidden_tile = tf.tile(Z_hidden, [n_batch, 1, 1, 1])

# initialize some weights
W_conv1 = weight_variable([L, L, 6, 4])
b_conv1 = bias_variable([4])

hidden_cat_obs = tf.concat(3, [Z_hidden_tile, in_obs])
print "hidden cat obs dim ", show_dim(hidden_cat_obs)

h_conv1 = tf.nn.sigmoid(conv2d(hidden_cat_obs, W_conv1) + b_conv1)
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)

# -------------------------------------------------------------------- recover z from embedding
W_conv1_z = weight_variable([L, L, 4, 1])
b_conv1_z = bias_variable([1])

z_conv1 = conv2d(h_conv1_drop, W_conv1_z) + b_conv1_z
print "conv 1 shape ", show_dim(z_conv1)
# z_pred = z_conv1 / norm_sum_tiled
z_pred = z_conv1

# ------------------------------------------------------------------------ simple cost functions
costt = tf.reduce_mean(tf.square(z_pred - out_z_loc))

# Minimize the mean squared errors (for the query inference)
# optimizer = tf.train.RMSPropOptimizer(0.001)
# train = optimizer.minimize(costt)

tvars = tf.trainable_variables()
grads = [tf.clip_by_value(grad, -1., 1.) for grad in tf.gradients(costt, tvars)]
optimizer = tf.train.RMSPropOptimizer(0.001)
train = optimizer.apply_gradients(zip(grads, tvars))

'''
# ------------------------------------------------------------------- make the query inference
# concat the input query with the hidden together
query_cat_hidden = tf.concat(3, [last_volvo, in_query_loc])
print "dim query_cat_hidden ", show_dim(query_cat_hidden)
 
# get some new weights making the inference
W_conv1_q = weight_variable([5, 5, 9, 20])
b_conv1_q = bias_variable([20])

W_conv2_q = weight_variable([5, 5, 20, 10])
b_conv2_q = bias_variable([10])

query_conv1 = tf.nn.relu(conv2d(query_cat_hidden, W_conv1_q) + b_conv1_q)
query_conv2 = tf.nn.relu(conv2d(query_conv1, W_conv2_q) + b_conv2_q)
 
print "query convolved ", show_dim(query_conv2)

# distill all that information into a single bit of True or False
W_fc1_q = weight_variable([20 * 20 * 10, 2])
b_fc1_q = bias_variable([2])
query_flat = tf.reshape(query_conv2, [-1, 20*20*10])
query_logit = tf.nn.relu(tf.matmul(query_flat, W_fc1_q) + b_fc1_q)
 
pred_lab = tf.nn.softmax(query_logit)
small_number = tf.constant(1e-10, shape=[n_batch, 2])
pred_lab = pred_lab + small_number
print "predicted label dim ", show_dim(pred_lab)

# ------------------------------------------------------------------------ training steps
# Minimize the mean squared errors (for the query inference)
xentropy =  -tf.reduce_sum(out_query_TF * tf.log(pred_lab))
optimizer = tf.train.RMSPropOptimizer(0.001)
train = optimizer.minimize(xentropy)
'''


# ------------------------------------------------------------------------------- running !

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# k_locs, k_weights, query_loc, query_TF, z_loc = gen_data(n_batch)
# feed_dic = gen_feed_dict(k_locs, k_weights, query_loc, query_TF, z_loc)
# print sess.run([costt], feed_dict=feed_dic)
# sess.run([train], feed_dict=feed_dic)
# print sess.run([costt], feed_dict=feed_dic)

# print sess.run([query_logit], feed_dict=feed_dic)
# print sess.run([pred_lab], feed_dict=feed_dic)

for i in range(5000001):
  obs, query_loc, query_TF, z_loc = gen_data(n_batch)
  feed_dic = gen_feed_dict(obs, query_loc, query_TF, z_loc)
  costpre = sess.run([costt], feed_dict=feed_dic)[0]
  sess.run([train], feed_dict=feed_dic)
  costpost = sess.run([costt], feed_dict=feed_dic)[0]
  print costpre, " ", costpost, " ", costpost - costpre < 0.0
  
  # do evaluation every 100 epochs
  if (i % 20 == 0):
    out_pred = sess.run(z_pred, feed_dict=feed_dic)
    # print "total sum ? ", np.sum(out_pred[0]), " ", np.sum(z_loc[0])
    draw(out_pred[0], "./drawings/{0}_run_pred.png".format(i/20))
    draw(z_loc[0], "./drawings/{0}_run_truth.png".format(i/20))
    draw_obs(obs[0], "./drawings/{0}_run_obs.png".format(i/20))


