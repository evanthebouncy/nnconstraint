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

# set up weights for input outputs!
in_k_coords = [tf.placeholder(tf.float32, [n_batch, 4]) for _ in range(K)]
in_k_weights = tf.placeholder(tf.float32, [n_batch, K])
in_query_coord = tf.placeholder(tf.float32, [n_batch, 2])
out_query_TF = tf.placeholder(tf.float32, [n_batch, 2])
out_z_coord = tf.placeholder(tf.float32, [n_batch, 2])
keep_prob = tf.placeholder(tf.float32)

def gen_feed_dict(k_coords, k_weights, query_coord, query_TF, z_coord):
  ret = {}
  for a, b in zip(in_k_coords, k_coords):
    ret[a] = b
  ret[in_k_weights] = k_weights
  ret[in_query_coord] = query_coord
  ret[out_query_TF] = query_TF
  ret[out_z_coord] = z_coord
  ret[keep_prob] = 0.7
  return ret


# --------------------------------------------------------------------- initial hidden state Z
# set up weights for input outputs!
enc_hidden = tf.Variable(tf.truncated_normal([1, 1000], stddev=0.1))
print "initial hidden dim ", show_dim(enc_hidden)
# enc_hidden_tile = tf.tile(enc_hidden, [n_batch]), [n_batch, 64]) for ww in unpacked_w]
enc_hidden_tile = tf.tile(enc_hidden, [n_batch, 1])
print "tiled hidden dim ", show_dim(enc_hidden_tile)

# --------------------------------------------------------------------- convolve in the observations

# initialize some weights
W_conv1 = weight_variable([1000+4, 1000])
b_conv1 = bias_variable([1000])

enc_hiddens = []
hidden_rollin = enc_hidden_tile
for i in range(K):
  print "volvoing input ", i
  obs_in = in_k_coords[i]
  print "input dim ", show_dim(obs_in)
  # concatenate the hidden with the input into a joint channal
  hidden_cat_input = tf.concat(1, [hidden_rollin, obs_in])
  print "concat dim of hidden and input ", show_dim(hidden_cat_input)
  # convolve them into the new hidden representation 
  h_conv1 = tf.nn.sigmoid(tf.matmul(hidden_cat_input, W_conv1) + b_conv1)
  h_conv2 = tf.nn.dropout(h_conv1, keep_prob)

  hidden_rollin = h_conv2
  enc_hiddens.append(hidden_rollin)
  print "rollin dim after takin in inputs ", show_dim(hidden_rollin)

print "hidden shape ", show_dim(enc_hiddens)


# --------------------------------------------------------------------------- error in all steps
inv_W = weight_variable([1000,2])
inv_b = weight_variable([2])

z_preds = [tf.matmul(volvoo, inv_W) + inv_b for volvoo in enc_hiddens]
print "pred shape ", show_dim(z_preds)

pred_costs = [tf.reduce_mean(tf.square(z_predd - out_z_coord), [1]) for z_predd in z_preds]

print "pred_costs shape ", show_dim(pred_costs)


# unpack the weights
unpacked_kw = tf.unpack(in_k_weights, axis=1)
print "unpacked kw shape ", show_dim(unpacked_kw)

pred_costs_weighted = zip(pred_costs, unpacked_kw)
weighted_costs = [x[0] * x[1] for x in pred_costs_weighted]
costt = tf.reduce_mean(tf.pack(weighted_costs))

print show_dim(costt)

# ------------------------------------------------------------------------ training steps
# Minimize the mean squared errors (for the query inference)
# costt = tf.reduce_mean(tf.square(z_pred - out_z_coord))
# optimizer = tf.train.RMSPropOptimizer(0.001)
# train = optimizer.minimize(costt)

tvars = tf.trainable_variables()
grads = [tf.clip_by_value(grad, -1., 1.) for grad in tf.gradients(costt, tvars)]
optimizer = tf.train.RMSPropOptimizer(0.0001)
train = optimizer.apply_gradients(zip(grads, tvars))

'''
# ------------------------------------------------------------------- make the query inference
# concat the input query with the hidden together
query_cat_hidden = tf.concat(3, [last_volvo, in_query_coord])
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

for i in range(5000001):
  k_coords, k_weights, query_coord, query_TF, z_coord = gen_data(n_batch)
  feed_dic = gen_feed_dict(k_coords, k_weights, query_coord, query_TF, z_coord)
  cost_pre = sess.run([costt], feed_dict=feed_dic)[0]
  sess.run([train], feed_dict=feed_dic)
  cost_post = sess.run([costt], feed_dict=feed_dic)[0]
  print cost_pre, " ", cost_post, " ", True if cost_post < cost_pre else False
  
  # do evaluation every 100 epochs
  if (i % 10 == 0):
    out_z_preds = sess.run(z_preds, feed_dict=feed_dic)
    out_z_batch0 = [xx[0] for xx in out_z_preds]
    k_weigh_batch0 = k_weights[0]
    zippo = zip(out_z_batch0, k_weigh_batch0)
    relevant_out = filter(lambda x: x[1] == 1.0, zippo)
    print "\n\n================================="
    print "true lable ", z_coord[0]
    print "guesses ", relevant_out


