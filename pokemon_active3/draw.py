import numpy as np
import numpy
import matplotlib.pylab as plt
# m = [[0.0, 1.47, 2.43, 3.44, 1.08, 2.83, 1.08, 2.13, 2.11, 3.7], [1.47, 0.0, 1.5,     2.39, 2.11, 2.4, 2.11, 1.1, 1.1, 3.21], [2.43, 1.5, 0.0, 1.22, 2.69, 1.33, 3.39, 2.15, 2.12, 1.87], [3.44, 2.39, 1.22, 0.0, 3.45, 2.22, 4.34, 2.54, 3.04, 2.28], [1.08, 2.11, 2.69, 3.45, 0.0, 3.13, 1.76, 2.46, 3.02, 3.85], [2.83, 2.4, 1.33, 2.22, 3.13, 0.0, 3.83, 3.32, 2.73, 0.95], [1.08, 2.11, 3.39, 4.34, 1.76, 3.83, 0.0, 2.47, 2.44, 4.74], [2.13, 1.1, 2.15, 2.54, 2.46, 3.32, 2.47, 0.0, 1.78, 4.01], [2.11, 1.1, 2.12, 3.04, 3.02, 2.73, 2.44, 1.78, 0.0, 3.57], [3.7, 3.21, 1.87, 2.28, 3.85, 0.95, 4.74, 4.01, 3.57, 0.0]]

def draw(m, name):
  matrix = m
  orig_shape = numpy.shape(matrix)
  # lose the channel shape in the end of orig_shape
  new_shape = orig_shape[:-1] 
#  print new_shape
  matrix = numpy.reshape(matrix, new_shape)
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_aspect('equal')
  plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
  # plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
  plt.colorbar()
  plt.savefig(name)

def draw_pred(m, name, true_lab, pred_lab):
  true_color = m.max() if true_lab[0] > true_lab[1] else m.min()
  pred_color = m.max() if pred_lab[0] > pred_lab[1] else m.min()
  m[0][0][0] = true_color
  m[0][1][0] = true_color
  draw(m, name)

def draw_obs(obs, name):
  splitted = np.split(obs, 2, axis=2)
  ret = splitted[0] -1.0 * splitted[1]
  draw(ret, name)

# def draw_obs(obs, name):
#   ret_shape = [20, 20, 1]
#   ret = numpy.zeros(shape=ret_shape)
#   for ob in obs:
#     if ob.max() > 0.0:
#       idxx = numpy.unravel_index(ob.argmax(), ob.shape)
#       ret[idxx[0]][idxx[1]] = 1.0
#     if ob.min() < 0.0:
#       idxx = numpy.unravel_index(ob.argmin(), ob.shape)
#       ret[idxx[0]][idxx[1]] = -1.0
#   draw(ret, name)

