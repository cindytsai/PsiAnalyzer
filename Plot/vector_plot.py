from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

VelRota = np.load("../Data/VelRota.npz")
VelLong = np.load("../Data/VelLong.npz")
field_shape = VelRota["x"].shape

X, Y, Z = np.meshgrid(np.arange(field_shape[0]), np.arange(field_shape[1]), np.arange(field_shape[2]), indexing='ij')

#ax.quiver(X[0:25, 0:25, 0:25], Y[0:25, 0:25, 0:25], Z[0:25, 0:25, 0:25], 
#           1e20*VelRota["x"][0:25, 0:25, 0:25],  1e20*VelRota["y"][0:25, 0:25, 0:25], 1e20*VelRota["z"][0:25, 0:25, 0:25], length=1)
#plt.show()

#ax.quiver(X[0:15, 0:15, 0:15], Y[0:15, 0:15, 0:15], Z[0:15, 0:15, 0:15], 
#        1e21*VelRota["x"][0:15, 0:15, 0:15],  1e21*VelRota["y"][0:15, 0:15, 0:15], 1e21*VelRota["z"][0:15, 0:15, 0:15], length=0.8, normalize=True, cmap='jet', lw=0.5)
#plt.show()


ax.quiver(X[0:15, 0:15, 0:15], Y[0:15, 0:15, 0:15], Z[0:15, 0:15, 0:15], 
        1e21*VelLong["x"][0:15, 0:15, 0:15],  1e21*VelLong["y"][0:15, 0:15, 0:15], 1e21*VelLong["z"][0:15, 0:15, 0:15], length=0.8, normalize=True, cmap='jet', lw=0.5)
plt.show()


