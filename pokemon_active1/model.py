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
in_obs = tf.placeholder(tf.float32, [n_batch, L, L, 1])
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

# initialize some weights
W_conv1 = weight_variable([8, 8, 1, 4])
b_conv1 = bias_variable([4])

h_conv1 = tf.nn.sigmoid(conv2d(in_obs, W_conv1) + b_conv1)
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)

flat = tf.reshape(h_conv1_drop, [n_batch, L*L*4])
W_flat = weight_variable([L*L*4, 40])
b_flat = bias_variable([40])
obs_flat = tf.nn.sigmoid(tf.matmul(flat, W_flat) + b_flat)
obs_embed = tf.nn.dropout(obs_flat, keep_prob)

# -------------------------------------------------------------------- recover z from embedding
# initialize some weights
W_unflat = weight_variable([40, L*L*1])
b_unflat = bias_variable([L*L*1])
z_unflat = tf.matmul(obs_embed, W_unflat)+b_unflat
z_unflat = tf.reshape(z_unflat, [n_batch,L,L,1])
print "unflat dim ", show_dim(z_unflat)

z_pred = z_unflat

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
  print sess.run([costt], feed_dict=feed_dic),
  sess.run([train], feed_dict=feed_dic)
  print sess.run([costt], feed_dict=feed_dic)
  
  # do evaluation every 100 epochs
  if (i % 50 == 0):
    out_pred = sess.run(z_pred, feed_dict=feed_dic)
    # print "total sum ? ", np.sum(out_pred[0]), " ", np.sum(z_loc[0])
    draw(out_pred[0], "./drawings/{0}_run_pred.png".format(i/50))
    draw(z_loc[0], "./drawings/{0}_run_truth.png".format(i/50))
    draw(obs[0], "./drawings/{0}_run_obs.png".format(i/50))


