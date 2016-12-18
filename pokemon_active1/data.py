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
      grid_coord_i = float(i)
      grid_coord_j = float(j)
      ret[i][j][0] = np.exp(-dist((grid_coord_i, grid_coord_j), coord))

  ssums = ret.sum()
  ret = ret / ssums
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
  Zx = np.random.randint(2,ll-2)
  Zy = np.random.randint(2,ll-2)
  return Zx, Zy

def gen_X(Z, ll = L):
  ll = float(ll)
  Xx = np.random.randint(2,ll-2)
  Xy = np.random.randint(2,ll-2)
  X = (Xx, Xy)
  if dist(Z,X) < ll / 3.0:
    return X, [1.0, 0.0]
  else:
    return X, [0.0, 1.0]

def gen_obs(Z, ob_num, ll=L):
  obs = [gen_X(Z) for _ in range(ob_num)]
  ret_shape = [ll, ll, 1]
  ret = numpy.zeros(shape=ret_shape)
  for ob, ob_lab in obs:
    multi = 1.0 if ob_lab[0] == 1.0 else -1.0
    for i in range(ll):
      for j in range(ll):
        grid_coord_i = float(i)
        grid_coord_j = float(j)
        ret[i][j][0] += np.exp(-dist((grid_coord_i, grid_coord_j), ob)) * multi

  return ret
  
# data of the form of
#    obs: an observation image
#    query_loc: the new query location
#    query_TF: the TF for that particular new query
#    z_loc: the location for the hidden state Z
# all variables are a list of tensors of dimention [n_batch x ...]   
# loc is an image of dimention L x L x 1
def gen_data(n_batch = n_batch, K=K):
  # tensor shape [batch x loc]
  obs = []
  # tensor shape [batch x loc]
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
    # for each batch, decide how many sample observations we want to draw
    k = numpy.random.randint(1,K)
    _obs = gen_obs(Z_coord, k)
    obs.append(_obs)

    z_loc.append(_z_loc)

  return np.array(obs, np.float32),\
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

