import tensorflow as tf
import numpy as np
from data import *
import os

def draw_batch0(draw_z_preds, draw_z_obs, draw_z_truth, draw_quests, draw_idx):
  draw_prefix = "./drawings/"

  # draw preds at all inputs
  z_bach0s = [z_pred[0] for z_pred in draw_z_preds]
  for i, zz in enumerate(z_bach0s):
    draw(zz, draw_prefix + "{0}/z_pred_{1}.png".format(draw_idx, i))

  # draw obs
  ob_bach0s = [draw_z_ob[0] for draw_z_ob in draw_z_obs]
  draw_obs(ob_bach0s, draw_prefix + "{0}/obs.png".format(draw_idx))

  # draw quests
  quest_bach0s = [draw_quest[0] for draw_quest in draw_quests]
  for i, quest in enumerate(quest_bach0s):
    draw(quest, draw_prefix + "{0}/quest_{1}.png".format(draw_idx, i))

  # draw truth
  draw(draw_z_truth[0], draw_prefix + "{0}/z_truth.png".format(draw_idx, i))

def write_bach0(write_preds, write_truths, draw_idx):
  draw_prefix = "./drawings/"
  to_write = ""
  for ppp, ooo in zip(write_preds, write_truths):
    to_write += str((ppp,ooo)) 
    ppp = [1.0, 0.0] if ppp[0] > ppp[1] else [0.0, 1.0]
    ooo = list(ooo)
    to_write += str( ppp == ooo ) + "\n"

  filename = draw_prefix+"{0}/pred.txt".format(draw_idx)

  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

  fd = open(filename, "w")
  fd.write(to_write)
  fd.close()


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
in_query_loc = tf.placeholder(tf.float32, [n_batch, L, L, 1])
out_query_TF = tf.placeholder(tf.float32, [n_batch, 2])
out_z_loc = tf.placeholder(tf.float32, [n_batch, L, L, 1])
keep_prob = tf.placeholder(tf.float32)

# def gen_feed_dict(k_locs, k_TFs, k_weights, query_loc, query_TF, z_loc):
def gen_feed_dict(k_locs, query_loc, query_TF, z_loc):
  ret = {}
  for a, b in zip(in_k_locs, k_locs):
    ret[a] = b
  ret[in_query_loc] = query_loc
  ret[out_query_TF] = query_TF
  ret[out_z_loc] = z_loc
  ret[keep_prob] = 0.7
  return ret

# --------------------------------------------------------------------- separate training vars
# vars used for inversion
V_inv = []
# vars used for prediction
V_pred = []

# --------------------------------------------------------------------- initial hidden state Z
# set up weights for input outputs!
Z_hidden = tf.Variable(tf.truncated_normal([1, L, L, 4], stddev=0.1))
V_inv.append(Z_hidden)
V_pred.append(Z_hidden)
print "initial hidden dim ", show_dim(Z_hidden)
# Z_hidden_tile = tf.tile(Z_hidden, [n_batch]), [n_batch, 64]) for ww in unpacked_w]
Z_hidden_tile = tf.tile(Z_hidden, [n_batch, 1, 1, 1])
print "tiled hidden dim ", show_dim(Z_hidden_tile)

# --------------------------------------------------------------------- convolve in the observations

# initialize some weights
W_conv1 = weight_variable([15, 15, 6, 4])
b_conv1 = bias_variable([4])
V_inv += [W_conv1, b_conv1]
V_pred += [W_conv1, b_conv1]

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
  h_conv1 = tf.nn.sigmoid(conv2d(hidden_cat_input, W_conv1) + b_conv1)
  h_conv2 = tf.nn.dropout(h_conv1, keep_prob)

  hidden_rollin = h_conv2
  Z_hiddens.append(hidden_rollin)
  print "rollin dim after takin in inputs ", show_dim(hidden_rollin)

# --------------------------------------------------------------------- predicting the z
# initialize some weights
W_conv1_z = weight_variable([15, 15, 4, 1])
b_conv1_z = bias_variable([1])
V_inv += [W_conv1_z, b_conv1_z]

z_preds = [conv2d(volvoo, W_conv1_z) + b_conv1_z for volvoo in Z_hiddens]

# cost for predicting the z
pred_costs = [tf.reduce_sum(tf.square(z_predd - out_z_loc), [1,2,3]) for z_predd in z_preds]

print "pred_costs shape ", show_dim(pred_costs)

z_preds_cost = tf.reduce_mean(tf.pack(pred_costs))

print "z_pred_cost shape ", show_dim(z_preds_cost)

# -------------------------------------------------------------------------- predicting ob
query_cat_hiddens = [tf.concat(3, [volvoo, in_query_loc]) for volvoo in Z_hiddens]
print "query_cat_hidden shape ", show_dim(query_cat_hiddens)
# get some new weights making the inference
W_conv1_q = weight_variable([15, 15, 5, 4])
b_conv1_q = bias_variable([4])
V_pred += [W_conv1_q, b_conv1_q]

query_volvos = [tf.nn.sigmoid(conv2d(q_cat_h, W_conv1_q) + b_conv1_q) for q_cat_h in query_cat_hiddens]
print "query volvos dim ", show_dim(query_volvos)

W_fc1_q = weight_variable([20 * 20 * 4, 2])
b_fc1_q = bias_variable([2])
V_pred += [W_fc1_q, b_fc1_q]

flat_query_volvos = [tf.reshape(q_v, [-1, 20*20*4]) for q_v in query_volvos]
query_logits = [tf.nn.relu(tf.matmul(f_qv, W_fc1_q) + b_fc1_q) for f_qv in flat_query_volvos]
 
pred_labs = [tf.nn.softmax(query_logit) for query_logit in query_logits]
small_number = tf.constant(1e-10, shape=[n_batch, 2])
pred_labs = [pred_lab + small_number for pred_lab in pred_labs]
print "predicted labels dim ", show_dim(pred_labs)

# xentrop errors for ob prediction
xentropys =  sum([-tf.reduce_sum(out_query_TF * tf.log(pred_lab)) for pred_lab in pred_labs])
print "xentropys shape ", show_dim(xentropys)

# ------------------------------------------------------------------- generate question
# initialize some weights
W_conv1_quest = weight_variable([15, 15, 4, 1])
b_conv1_quest = bias_variable([1])
V_quest = [W_conv1_quest, b_conv1_quest]

q_quests = [conv2d(volvoo, W_conv1_quest) + b_conv1_quest for volvoo in Z_hiddens]

# make question have maximum confusion
quests_hidden_zip = zip(q_quests, Z_hiddens)
quest_cat_hiddens = [tf.concat(3, list(zippy)) for zippy in quests_hidden_zip]
quest_volvos = [tf.nn.sigmoid(conv2d(q_cat_h, W_conv1_q) + b_conv1_q) for q_cat_h in quest_cat_hiddens]

flat_quest_volvos = [tf.reshape(q_v, [-1, 20*20*4]) for q_v in quest_volvos]
quest_logits = [tf.nn.relu(tf.matmul(f_qv, W_fc1_q) + b_fc1_q) for f_qv in flat_quest_volvos]
 
quest_labs = [tf.nn.softmax(quest_logit) for quest_logit in quest_logits]

confusion_lab = tf.tile(tf.constant([[0.5, 0.5]]), [50, 1])
quest_xentropys =  sum([tf.reduce_sum(tf.square(confusion_lab - quest_lab)) for quest_lab in quest_labs])

# ------------------------------------------------------ discriminant network for question
W_conv1_dis = weight_variable([15, 15, 1, 1])
b_conv1_dis = bias_variable([1])
W_fc1_dis = weight_variable([20 * 20 * 1, 2])
b_fc1_dis = bias_variable([2])
V_dis = [W_conv1_dis, b_conv1_dis, W_fc1_dis, b_fc1_dis]

# ----------------- generate true on real observes
x_dis_conv1 = tf.nn.sigmoid(conv2d(in_query_loc, W_conv1_dis) + b_conv1_dis)
x_dis_flat = tf.reshape(x_dis_conv1, [50, 20*20*1])

x_dis_pred = tf.nn.softmax(tf.matmul(x_dis_flat, W_fc1_dis) + b_fc1_dis)
# demand x_dis_pred to be TRUE
true_lab = tf.tile(tf.constant([[1.0, 0.0]]), [50, 1])
print "true_lab shape ", true_lab.get_shape()
cost_dis_true = tf.reduce_mean(tf.square(true_lab - x_dis_pred))
# ----------------- generate false on pretended observers
fake_dis_conv1s = [tf.nn.sigmoid(conv2d(x, W_conv1_dis) + b_conv1_dis) for x in q_quests]
fake_dis_flats = [tf.reshape(fake_dis_conv1, [50, 20*20*1]) for fake_dis_conv1 in fake_dis_conv1s]

fake_dis_preds = [tf.nn.softmax(tf.matmul(fake_dis_flat, W_fc1_dis) + b_fc1_dis)\
                  for fake_dis_flat in fake_dis_flats]
# demand fake_dis_pred to be FALSE
false_labs = [tf.tile(tf.constant([[0.0, 1.0]]), [50, 1]) for _ in range(K)]
print "false_labs shape ", show_dim(false_labs)
zipppp = zip(fake_dis_preds, false_labs)
cost_dis_false = sum([tf.reduce_mean(tf.square(x[0] - x[1])) for x in zipppp])

dis_cost = cost_dis_true + cost_dis_false

# --------------------------------------- wrap up the quest gen network to account for dis cost
# demand fake_dis_pred to be TRUE
true_labs = [tf.tile(tf.constant([[1.0, 0.0]]), [50, 1]) for _ in range(K)]
print "true_labs shape ", show_dim(true_labs)
zipppp = zip(fake_dis_preds, true_labs)
cost_dis_fake_true = sum([tf.reduce_mean(tf.square(x[0] - x[1])) for x in zipppp])

quest_cost = quest_xentropys + cost_dis_fake_true

# ------------------------------------------------------------------------ training steps

optimizer = tf.train.RMSPropOptimizer(0.001)
# training for inversion
inv_grads = [tf.clip_by_value(grad, -5., 5.) for grad in tf.gradients(z_preds_cost, V_inv)]
inv_train = optimizer.apply_gradients(zip(inv_grads, V_inv))
# training for prediction
pred_grads = [tf.clip_by_value(grad, -5., 5.) for grad in tf.gradients(xentropys, V_pred)]
pred_train = optimizer.apply_gradients(zip(pred_grads, V_pred))

# training for question
quest_grads = [tf.clip_by_value(grad, -5., 5.) for grad in tf.gradients(quest_cost, V_quest)]
quest_train = optimizer.apply_gradients(zip(quest_grads, V_quest))

# # training for question (dis)
# quest_dis_grads = [tf.clip_by_value(grad, -5., 5.) for grad in tf.gradients(cost_dis_fake_true, V_quest)]
# quest_dis_train = optimizer.apply_gradients(zip(quest_grads, V_quest))

# training for dis
dis_grads = [tf.clip_by_value(grad, -5., 5.) for grad in tf.gradients(dis_cost, V_dis)]
dis_train = optimizer.apply_gradients(zip(dis_grads, V_dis))

# ------------------------------------------------------------------------------- running !

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

for i in range(5000001):
  k_locs, query_loc, query_TF, z_loc = gen_data(n_batch)
  feed_dic = gen_feed_dict(k_locs, query_loc, query_TF, z_loc)
  # do evaluation every 100 epochs
  if (i % 20 == 0):

    query_preds = sess.run(pred_labs, feed_dict=feed_dic)
    q_preds_bach0 = [qpp[0] for qpp in query_preds]
    q_preds_truth_bach0 = [query_TF[0] for _ in range(K)]
    write_bach0(q_preds_bach0, q_preds_truth_bach0, i/20)

    out_z_preds = sess.run(z_preds, feed_dict=feed_dic)
    out_quests = sess.run(q_quests, feed_dict=feed_dic)
    draw_batch0(out_z_preds, 
                k_locs,
                z_loc,
                out_quests,
                i / 20)


  print "z pred cost ", sess.run([z_preds_cost], feed_dict=feed_dic),
  sess.run([inv_train], feed_dict=feed_dic)
  print sess.run([z_preds_cost], feed_dict=feed_dic)
  
  print "prediction cost ", sess.run([xentropys], feed_dict=feed_dic),
  sess.run([pred_train], feed_dict=feed_dic)
  print sess.run([xentropys], feed_dict=feed_dic)

  print "quest cost ", sess.run([quest_cost], feed_dict=feed_dic),
  sess.run([quest_train], feed_dict=feed_dic)
  print sess.run([quest_cost], feed_dict=feed_dic)

#   print "quest confusion cost ", sess.run([quest_xentropys], feed_dict=feed_dic),
#   sess.run([quest_train], feed_dict=feed_dic)
#   print sess.run([quest_xentropys], feed_dict=feed_dic)
# 
#   print "quest dis cost ", sess.run([cost_dis_fake_true], feed_dict=feed_dic),
#   sess.run([quest_dis_train], feed_dict=feed_dic)
#   print sess.run([cost_dis_fake_true], feed_dict=feed_dic)

  print "dis cost ", sess.run([dis_cost], feed_dict=feed_dic),
  sess.run([dis_train], feed_dict=feed_dic)
  print sess.run([dis_cost], feed_dict=feed_dic)

