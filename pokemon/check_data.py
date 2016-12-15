from data import *

k_locs, k_TFs, k_weights, query_loc, query_TF, z_loc = gen_data(3)

print "batch size = 3"
print "observation number up to = 10"
print show_dim(k_locs)
print show_dim(k_TFs)
print np.shape(k_weights)
print np.shape(query_loc)
print np.shape(query_TF)
print np.shape(z_loc)


# check some shapes by printing
print k_weights
for k_TF in k_TFs:
  print k_TF

for zz_loc in z_loc:
  draw(zz_loc)

