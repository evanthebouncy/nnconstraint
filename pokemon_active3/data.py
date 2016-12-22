import numpy as np
from draw import *

K = 16
L = 20
n_batch = 50

# ------------------------------------------------------------------ helpers
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
      ret[i][j][0] = np.exp(-dist((grid_coord_i, grid_coord_j), coord) / 2.0)

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
  ret_shape = [ll, ll, 2]
  ret = numpy.zeros(shape=ret_shape)
  for ob, ob_lab in obs:
    idxx = 0 if ob_lab[0] == 1.0 else 1
    for i in range(ll):
      for j in range(ll):
        grid_coord_i = float(i)
        grid_coord_j = float(j)
        ret[i][j][idxx] = max(np.exp(-dist((grid_coord_i, grid_coord_j), ob) / 2.0),
                              ret[i][j][idxx])

  return ret, obs
  
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

  xtra_info = {}

  for bb in range(n_batch):
    xtra_info[bb] = {}
    # generate a hidden variable Z for each batch
    Z_coord = gen_Z()
    _z_loc = coord_2_loc(Z_coord)
    # for each batch, decide how many sample observations we want to draw
    k = numpy.random.randint(K/2,K)

    _obs, wololo = gen_obs(Z_coord, k)
    obs.append(_obs)

    # generate and add query location
    _query_coord, _query_TF = gen_X(Z_coord)

    _query_loc = coord_2_loc(_query_coord)
    query_loc.append(_query_loc)
    query_TF.append(_query_TF)
    

    z_loc.append(_z_loc)
    xtra_info[bb]["z_coord"] = Z_coord

  return np.array(obs, np.float32),\
         np.array(query_loc, np.float32), \
         np.array(query_TF, np.float32), \
         np.array(z_loc, np.float32), \
         xtra_info
      
# dat_in, dat_out = gen_data()
# print np.shape(dat_in)
# print dat_out
    

#sqs = gen_squares()
#draw(gen_squares())
# Z = gen_Z()
# print Z
# draw(coord_2_loc(Z))

