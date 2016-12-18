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

# set up weights for input outputs!
in_k_locs = [tf.placeholder(tf.float32, [n_batch, L, L, 2]) for _ in range(K)]
# in_k_TFs = [tf.placeholder(tf.float32, [n_batch, 2]) for _ in range(K)]
in_k_weights = tf.placeholder(tf.float32, [n_batch, K])
in_query_loc = tf.placeholder(tf.float32, [n_batch, L, L, 1])
out_query_TF = tf.placeholder(tf.float32, [n_batch, 2])
out_z_loc = tf.placeholder(tf.float32, [n_batch, L, L, 1])

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
  return ret


# --------------------------------------------------------------------------- embed observations
W_conv1 = weight_variable([5, 5, 2, 8])
b_conv1 = bias_variable([8])

h_conv1s = [tf.nn.relu(conv2d(x, W_conv1) + b_conv1) for x in in_k_locs]
h_pool1s = [max_pool_2x2(h_conv1) for h_conv1 in h_conv1s]


print show_dim(h_conv1s)
print show_dim(h_pool1s)

W_conv2 = weight_variable([5, 5, 8, 8])
b_conv2 = bias_variable([8])

h_conv2s = [tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) for h_pool1 in h_pool1s]
h_pool2s = [max_pool_2x2(h_conv2) for h_conv2 in h_conv2s]

W_fc1 = weight_variable([5 * 5 * 8, 64])
b_fc1 = bias_variable([64])

h_pool2_flats = [tf.reshape(h_pool2, [-1, 5*5*8]) for h_pool2 in h_pool2s]
h_fc1s = [tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) for h_pool2_flat in h_pool2_flats]

print show_dim(h_fc1s)
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# concat the convolved input query with the input TF value for that query, and pass is through
# another RELU layer
# fc1_with_labs = [tf.concat(1, xx) for xx in zip(h_fc1s, in_k_TFs)]
# print show_dim(fc1_with_labs)

W_fc2 = weight_variable([64, 64])
b_fc2 = bias_variable([64])

# obs_embeds = [tf.nn.relu(tf.matmul(fc1_with_lab, W_fc2) + b_fc2) for\
#                   fc1_with_lab in fc1_with_labs]
obs_embeds = [tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2) for\
                  fc1 in h_fc1s]
print show_dim(obs_embeds)

# unpack the weights
unpacked_w = tf.unpack(in_k_weights, axis=1)
# reshape it to match the embedded thing for each input
unpacked_w = [tf.reshape(tf.tile(ww, [64]), [n_batch, 64]) for ww in unpacked_w]
print type(unpacked_w)
print show_dim(unpacked_w)
zippp = zip(unpacked_w, obs_embeds)
# weigh each embeding
weighted_embs = [x[0] * x[1] for x in zippp]
# sum the embedings together
obs_embed = sum(weighted_embs)
print show_dim(obs_embed)

# ----------------------------------------------------------------------------- readin query
# use shared weights from the first layer of the observation embedding
W_conv1_q = weight_variable([5, 5, 1, 8])
b_conv1_q = bias_variable([8])

W_conv2_q = weight_variable([5, 5, 8, 8])
b_conv2_q = bias_variable([8])

W_fc1_q = weight_variable([5 * 5 * 8, 64])
b_fc1_q = bias_variable([64])

query_conv1 = tf.nn.relu(conv2d(in_query_loc, W_conv1_q) + b_conv1_q)
query_pool1 = max_pool_2x2(query_conv1)

query_conv2 = tf.nn.relu(conv2d(query_pool1, W_conv2_q) + b_conv2_q)
query_pool2 = max_pool_2x2(query_conv2)

query_pool2_flat = tf.reshape(query_pool2, [-1, 5*5*8])
query_emb = tf.nn.relu(tf.matmul(query_pool2_flat, W_fc1_q) + b_fc1_q)

print show_dim(query_emb)

# ------------------------------------------------------------------- make the query inference
# first we concatenate the query_emb with the obs_emb
conc_q_obs = tf.concat(1, [query_emb, obs_embed])
print show_dim(conc_q_obs)
W_q_infer = weight_variable([128, 2])
b_q_infer = bias_variable([2])

pred_lab = tf.nn.softmax(tf.matmul(conc_q_obs, W_q_infer) + b_q_infer) 
small_number = tf.constant(1e-10, shape=[n_batch, 2])
pred_lab = pred_lab + small_number
print show_dim(pred_lab)

# ------------------------------------------------------------------ make the z inference (lata)
# 

# ------------------------------------------------------------------------ training steps
# Minimize the mean squared errors (for the query inference)

# loss = tf.reduce_mean(tf.square(pred_lab - out_query_TF))
xentropy =  -tf.reduce_sum(out_query_TF * tf.log(pred_lab))

# optimizer = tf.train.GradientDescentOptimizer(0.01)
optimizer = tf.train.RMSPropOptimizer(0.002)

train = optimizer.minimize(xentropy)



# ------------------------------------------------------------------------------- running !

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

k_locs, k_weights, query_loc, query_TF, z_loc = gen_data(n_batch)
feed_dic = gen_feed_dict(k_locs, k_weights, query_loc, query_TF, z_loc)
print sess.run([xentropy], feed_dict=feed_dic)
sess.run([train], feed_dict=feed_dic)
print sess.run([xentropy], feed_dict=feed_dic)
# print sess.run([y_conv], feed_dict={x: dat, y_true: lab, keep_prob: 1.0})

for i in range(5000001):
  # k_locs, k_TFs, k_weights, query_loc, query_TF, z_loc = gen_data(n_batch)
  k_locs, k_weights, query_loc, query_TF, z_loc = gen_data(n_batch)
  feed_dic = gen_feed_dict(k_locs, k_weights, query_loc, query_TF, z_loc)
  print sess.run([xentropy], feed_dict=feed_dic),
  sess.run([train], feed_dict=feed_dic)
  print sess.run([xentropy], feed_dict=feed_dic)
  
  # do evaluation every 100 epochs
  if (i % 100 == 0):
    print "evaluating on some samples ... "
    pred_y = sess.run([pred_lab], feed_dict=feed_dic)
    for ppp, ooo in zip(list(pred_y[0]), list(query_TF)):
      print ppp, ooo, 
      ppp = [1.0, 0.0] if ppp[0] > ppp[1] else [0.0, 1.0]
      ooo = list(ooo)
      print ppp == ooo

  # continuously train at every epoch

