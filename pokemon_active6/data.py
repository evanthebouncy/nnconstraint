import numpy as np
from draw import *

K = 10
L = 20
n_batch = 50

# ------------------------------------------------------------------ helpers

# pad a list into its final length and also turn it into a numpy array
def pad_and_numpy(lst, final_length):
  # convert
  lst = [np.array(x, np.float32) for x in lst]
  length = len(lst)
  # crop if too long
  if length > final_length:
    return lst[:final_length]
  item_shape = lst[0].shape
  return lst + [np.zeros(item_shape, np.float32) for i in range(final_length - length)]

def dist(pt1, pt2):
  x1, y1 = pt1
  x2, y2 = pt2
  return np.sqrt(np.power(x1-x2,2) + np.power(y1-y2,2))

def coord_2_loc(coord, ll = L):
  ret = np.zeros([ll, ll, 1])
  for i in range(ll):
    for j in range(ll):
      grid_coord_i = i + 0.5
      grid_coord_j = j + 0.5
      ret[i][j][0] = np.exp(-dist((grid_coord_i, grid_coord_j), coord) / 1.0)

  # ssums = ret.sum()
  # ret = ret / ssums
  return ret

def consistent_z(obs, ll = L):
  def consistent(z_pt, obs):
    for ob_pt, ob_pred in obs:
      if dist(z_pt, ob_pt) > ll / 2.0 and ob_pred:
        return False
      if dist(z_pt, ob_pt) < ll / 2.0 and not ob_pred:
        return False
    return True
      
  ret = np.zeros([ll, ll, 1])
  for i in range(ll):
    for j in range(ll):
      grid_coord_i = i + 0.5
      grid_coord_j = j + 0.5
      ret[i][j][0] = 1.0 if consistent((grid_coord_i, grid_coord_j), obs) else 0.0

  ssums = ret.sum()
  ret = ret / (ssums + 0.001)
  return ret

def coord_2_loc_obs(coord, label, ll = L):
  idxx = 0 if label[0] == 1.0 else 1
  ret = np.zeros([ll, ll, 2])
  for i in range(ll):
    for j in range(ll):
      grid_coord_i = i + 0.5
      grid_coord_j = j + 0.5
      ret[i][j][idxx] = np.exp(-dist((grid_coord_i, grid_coord_j), coord) / 1.0)
  return ret

# show dimension of a data object (list of list or a tensor)
def show_dim(lst1):
  if hasattr(lst1, '__len__') and len(lst1) > 0:
    return [len(lst1), show_dim(lst1[0])]
  else:
    try:
      return lst1.get_shape()
    except:
      try:
        return lst1.shape
      except:
        return type(lst1)

# --------------------------------------------------------------- modelings
# generate the hidden state
def gen_Z(ll = L):
  x_coord = np.random.random() * (ll - 4.0) + 2.0
  y_coord = np.random.random() * (ll - 4.0) + 2.0
  return x_coord, y_coord

def gen_X(Z, ll = L):
  ll = float(ll)
  Xx = np.random.random() * (ll - 4.0) + 2.0
  Xy = np.random.random() * (ll - 4.0) + 2.0
  X = (Xx, Xy)
  if dist(Z,X) < ll / 3.0:
    return X, [1.0, 0.0]
  else:
    return X, [0.0, 1.0]
  
# data of the form of
#    k_locs: the k observation locations
#    k_TFs: the true/false value of these observations
#    k_weights: the average 1/k weight for each observation
#    query_loc: the new query location
#    query_TF: the TF for that particular new query
#    z_loc: the location for the hidden state Z
# all variables are a list of tensors of dimention [n_batch x ...]   
def gen_data(n_batch = n_batch, K=K):
  # LIST of length K (1 for each input)
  # each element of shape [batch x loc]
  k_locs = [[] for i in range(K)]
  # tensor shape [batch x K] (no more list after trying to join together)
  k_weights = []
  # tensor shape of [batch x loc]
  last_weights = []
  query_loc = []
  # tensor shape of [batch x 2] (2 classes, T or F)
  query_TF = []
  # tensor shape of [batch x loc]
  z_loc = []

  for bb in range(n_batch):
    # generate a hidden variable Z for each batch
    Z_coord = gen_Z()
    _z_loc = coord_2_loc(Z_coord)
    # generate and add query location
    _query_coord, _query_TF = gen_X(Z_coord)
    _query_loc = coord_2_loc(_query_coord)
    query_loc.append(_query_loc)
    query_TF.append(_query_TF)

    # for easier padding and such, generate these for each batch then mush them in
    # only the list of tensors need this treatment, otherwise just batch them in ok
    b_k_locs = []

    for _ in range(K):
      obs_x, obs_lab = gen_X(Z_coord)
      obs_x_loc = coord_2_loc_obs(obs_x, obs_lab)
      b_k_locs.append(obs_x_loc)

    b_k_locs = pad_and_numpy(b_k_locs, K) 
    for kkk in range(K):
      k_locs[kkk].append(b_k_locs[kkk]) 

    z_loc.append(_z_loc)

  return k_locs, \
         np.array(query_loc, np.float32), \
         np.array(query_TF, np.float32), \
         np.array(z_loc, np.float32)
      
# dat_in, dat_out = gen_data()
# print np.shape(dat_in)
# print dat_out
    

#sqs = gen_squares()
#draw(gen_squares())
# Z = gen_Z()
# print Z
# draw(coord_2_loc(Z))

