import numpy as np
from draw import *

K = 10
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
def gen_Z():
  x_coord = np.random.random()
  y_coord = np.random.random()
  return x_coord, y_coord

def gen_X(Z):
  Xx = np.random.random()
  Xy = np.random.random()
  X = (Xx, Xy)
  if dist(Z,X) < 1.0 / 3.0:
    return X, [1.0, 0.0]
  else:
    return X, [0.0, 1.0]
  
# data of the form of
# everything is of length k
#    k_ob_coords: the k observation coordations, along with TF values
#    query_coord: the k new query coordations
#    query_TF: the k TF for the query coords
#    z_coord: the coordation for the hidden state Z
#    k_weights: the average 1/k weight for each observation
# all variables are a list of tensors of dimention [n_batch x ...]   
def gen_data(n_batch = n_batch, K=K):
  # LIST of length K (1 for each input)
  # each element of shape [batch x coord]
  k_ob_coords = [[] for i in range(K)]
  # tensor shape [batch x K] (no more list after trying to join together)
  k_weights = []
  # tensor shape of [batch x coord]
  query_coord = []
  # tensor shape of [batch x 2] (2 classes, T or F)
  query_TF = []
  # tensor shape of [batch x coord]
  z_coord = []

  for bb in range(n_batch):
    # generate a hidden variable Z for each batch
    Z_coord = gen_Z()
    # generate and add query location
    _query_coord, _query_TF = gen_X(Z_coord)
    query_coord.append(_query_coord)
    query_TF.append(_query_TF)
    # for each batch, decide how many sample observations we want to draw
    k = np.random.randint(1,K)
    # then re-weight each input layer by the appropriate weight
    _k_weights = [0.0 for _ in range(K)]
    for haha in range(k):
      _k_weights[haha] = 1.0
    k_weights.append(_k_weights)

    # for easier padding and such, generate these for each batch then mush them in
    # only the list of tensors need this treatment, otherwise just batch them in ok
    b_k_locs = []
    for _ in range(k):
      obs_x, obs_lab = gen_X(Z_coord)
      obs_x_loc = list(obs_x) + obs_lab
      b_k_locs.append(obs_x_loc)

    b_k_locs = pad_and_numpy(b_k_locs, K) 
    for kkk in range(K):
      k_ob_coords[kkk].append(b_k_locs[kkk]) 

    z_coord.append(Z_coord)

  return k_ob_coords, \
         np.array(k_weights, np.float32), \
         np.array(query_coord, np.float32), \
         np.array(query_TF, np.float32), \
         np.array(z_coord, np.float32)
      
