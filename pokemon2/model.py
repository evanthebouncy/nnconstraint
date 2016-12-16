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
in_k_locs = [tf.placeholder(tf.float32, [n_batch, L, L, 2]) for _ in range(K)]
in_k_weights = tf.placeholder(tf.float32, [n_batch, K])
in_query_loc = tf.placeholder(tf.float32, [n_batch, L, L, 1])
out_query_TF = tf.placeholder(tf.float32, [n_batch, 2])
out_z_loc = tf.placeholder(tf.float32, [n_batch, L, L, 1])
keep_prob = tf.placeholder(tf.float32)

# def gen_feed_dict(k_locs, k_TFs, k_weights, query_loc, query_TF, z_loc):
def gen_feed_dict(k_locs, k_weights, query_loc, query_TF, z_loc):
  ret = {}
  for a, b in zip(in_k_locs, k_locs):
    ret[a] = b
#  for a, b in zip(in_k_TFs, k_TFs):
#    ret[a] = b
  ret[in_k_weights] = k_weights
  ret[in_query_loc] = query_loc
  ret[out_query_TF] = query_TF
  ret[out_z_loc] = z_loc
  ret[keep_prob] = 0.9
  return ret


# --------------------------------------------------------------------- initial hidden state Z
# set up weights for input outputs!
Z_hidden = tf.Variable(tf.truncated_normal([1, L, L, 1], stddev=0.1))
print "initial hidden dim ", show_dim(Z_hidden)
# Z_hidden_tile = tf.tile(Z_hidden, [n_batch]), [n_batch, 64]) for ww in unpacked_w]
Z_hidden_tile = tf.tile(Z_hidden, [n_batch, 1, 1, 1])
print "tiled hidden dim ", show_dim(Z_hidden_tile)

# --------------------------------------------------------------------- convolve in the observations

# initialize some weights
W_conv1 = weight_variable([5, 5, 3, 8])
b_conv1 = bias_variable([8])
W_conv2 = weight_variable([5, 5, 8, 1])
b_conv2 = bias_variable([1])

Z_hiddens = []
hidden_rollin = Z_hidden_tile
for i in range(K):
  print "volvoing input ", i
  obs_in = in_k_locs[i]
  print "input dim ", show_dim(obs_in)
  # concatenate the hidden with the input into a joint channal
  hidden_cat_input = tf.concat(3, [hidden_rollin, obs_in])
  print "concat dim of hidden and input ", show_dim(hidden_cat_input)
  # convolve them into the new hidden representation 
  h_conv1 = tf.nn.relu(conv2d(hidden_cat_input, W_conv1) + b_conv1)
  h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

  h_conv2 = tf.nn.dropout(h_conv2, keep_prob)

  hidden_rollin = h_conv2
  Z_hiddens.append(hidden_rollin)
  print "rollin dim after takin in inputs ", show_dim(hidden_rollin)


# ---------------------------------------------------------------------- use last volvo for inference
# unpack the weights
unpacked_w = tf.unpack(in_k_weights, axis=1)
# reshape it to match the dim for the hidden_volvos
unpacked_w = [tf.reshape(tf.tile(ww, [L * L]), [n_batch, L, L, 1]) for ww in unpacked_w]
print "dim umpacked weights ", show_dim(unpacked_w)
zippp = zip(unpacked_w, Z_hiddens)
# weigh each hidden volvo
weighted_volvos = [x[0] * x[1] for x in zippp]
# sum the hiddens together (i.e. output the last volvo)
last_volvo = sum(weighted_volvos)
print "last volvo dim ", show_dim(last_volvo)

# ------------------------------------------------------------------- make the query inference
# concat the input query with the hidden together
# query_cat_hidden = tf.concat(3, [last_volvo, in_query_loc])
# print "dim query_cat_hidden ", show_dim(query_cat_hidden)
#  
# # get some new weights making the inference
# W_conv1_q = weight_variable([5, 5, 2, 8])
# b_conv1_q = bias_variable([8])
# 
# W_conv2_q = weight_variable([5, 5, 8, 1])
# b_conv2_q = bias_variable([1])
# 
# query_conv1 = tf.nn.relu(conv2d(query_cat_hidden, W_conv1_q) + b_conv1_q)
# query_conv2 = tf.nn.relu(conv2d(query_conv1, W_conv2_q) + b_conv2_q)
#  
# print "query convolved ", show_dim(query_conv2)
# 
# # distill all that information into a single bit of True or False
# W_fc1_q = weight_variable([20 * 20 * 1, 2])
# b_fc1_q = bias_variable([2])
# query_flat = tf.reshape(query_conv2, [-1, 20*20*1])
# query_logit = tf.nn.relu(tf.matmul(query_flat, W_fc1_q) + b_fc1_q)
#  
# pred_lab = tf.nn.softmax(query_logit)
# small_number = tf.constant(1e-10, shape=[n_batch, 2])
# pred_lab = pred_lab + small_number
# print "predicted label dim ", show_dim(pred_lab)
#  
# # # ------------------------------------------------------------------ make the z inference (lata)
# # # 
# # 
# 
# # ------------------------------------------------------------------------ training steps
# # Minimize the mean squared errors (for the query inference)
# xentropy =  -tf.reduce_sum(out_query_TF * tf.log(pred_lab))
# optimizer = tf.train.RMSPropOptimizer(0.01)
# train = optimizer.minimize(xentropy)
# 
# 
# 
# ------------------------------------------------------------------------------- running !

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# k_locs, k_weights, query_loc, query_TF, z_loc = gen_data(n_batch)
# feed_dic = gen_feed_dict(k_locs, k_weights, query_loc, query_TF, z_loc)
# print sess.run([xentropy], feed_dict=feed_dic)
# sess.run([train], feed_dict=feed_dic)

# print sess.run([xentropy], feed_dict=feed_dic)
# print sess.run([query_logit], feed_dict=feed_dic)
# print sess.run([pred_lab], feed_dict=feed_dic)

# for i in range(100):
#   k_locs, k_weights, query_loc, query_TF, z_loc = gen_data(n_batch)
#   feed_dic = gen_feed_dict(k_locs, k_weights, query_loc, query_TF, z_loc)
#   print sess.run([xentropy], feed_dict=feed_dic),
#   sess.run([train], feed_dict=feed_dic)
#   print sess.run([xentropy], feed_dict=feed_dic)

for i in range(5000001):
  k_locs, k_weights, query_loc, query_TF, z_loc = gen_data(n_batch)
  feed_dic = gen_feed_dict(k_locs, k_weights, query_loc, query_TF, z_loc)
  print sess.run([xentropy], feed_dict=feed_dic),
  sess.run([train], feed_dict=feed_dic)
  print sess.run([xentropy], feed_dict=feed_dic)
  
  # do evaluation every 100 epochs
  if (i % 10 == 0):
    print "evaluating on some samples ... "
    pred_y = sess.run([pred_lab], feed_dict=feed_dic)
    lastt = sess.run(last_volvo, feed_dict=feed_dic)
    print lastt[0]
    draw(lastt[0])
    for ppp, ooo in zip(list(pred_y[0]), list(query_TF)):
      print ppp, ooo, 
      ppp = [1.0, 0.0] if ppp[0] > ppp[1] else [0.0, 1.0]
      ooo = list(ooo)
      print ppp == ooo


