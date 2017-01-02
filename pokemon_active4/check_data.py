from data import *

k_coords, k_weights, query_coord, query_TF, z_coord = gen_data(3)

print "batch size = 3"
print "observation number up to = 10"
print show_dim(k_coords)
print np.shape(k_weights)
print np.shape(query_coord)
print np.shape(query_TF)
print np.shape(z_coord)

# check some shapes by printing
print k_weights
print zip(query_coord, query_TF, z_coord)
