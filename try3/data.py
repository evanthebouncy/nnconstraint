import numpy as np
from draw import *

L = 20

# has a single channel
def gen_squares(ll = L):
  ret = np.zeros([ll, ll, 1])
  l = L / 5
  for i in range(15):
    x, y = np.random.randint(1, ll - l), np.random.randint(1, ll - l)
    # draw a rectangle starting at (x,y) as upper left corner   
    for ii in range(l):
      for jj in range(l):
        ret[x+ii][y+jj][0] = 1.0
  return ret

def gen_triangles(ll = L):
  ret = np.zeros([ll, ll, 1])
  l = L / 5
  for i in range(15):
    x, y = np.random.randint(1, ll - l), np.random.randint(1, ll - l)
    # draw a rectangle starting at (x,y) as upper left corner   
    for ii in range(l):
      for jj in range(l):
        if ii <= jj:
          ret[x+ii][y+jj][0] = 1.0
  return ret

def gen_data(n = 50):
  ret_input = []
  ret_output = []
  for i in range(n):
#    if np.random.random() > 0.5:
      ret_input.append(gen_squares())
      ret_output.append([1.0, 0.0])
#    else:
#      ret_input.append(gen_triangles())
#      ret_output.append([0.0, 1.0])

  return np.array(ret_input, np.float32), np.array(ret_output, np.float32)
      
# dat_in, dat_out = gen_data()
# print np.shape(dat_in)
# print dat_out
    

#sqs = gen_squares()
#draw(gen_squares())
# draw(gen_triangles())
