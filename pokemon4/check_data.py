from data import *

k_locs, k_weights, last_weights, query_loc, query_TF, z_loc = gen_data(3)

print "batch size = 3"
print "observation number up to = 10"
print show_dim(k_locs)
print np.shape(k_weights)
print np.shape(last_weights)
print np.shape(query_loc)
print np.shape(query_TF)
print np.shape(z_loc)

# check some shapes by printing
print k_weights
draw(z_loc[0], "check_data_z.png")
draw_obs([k_loc[0] for k_loc in k_locs], "check_data_obs.png")
