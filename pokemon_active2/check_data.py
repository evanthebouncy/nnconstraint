from data import *

obs, query_loc, query_TF, z_loc = gen_data(3)

print "batch size = 3"
print "observation number up to = 10"
print show_dim(obs)
print np.shape(query_loc)
print np.shape(query_TF)
print np.shape(z_loc)

# check some shapes by printing
draw(z_loc[0], "check_data_z.png")
draw_obs(obs[0], "check_data_obs.png")
draw(query_loc[0], "check_data_query.png")
