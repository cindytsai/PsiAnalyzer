from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

VelRota = np.load("../Data/VelRota.npz")
VelLong = np.load("../Data/VelLong.npz")
field_shape = VelRota["x"].shape

X, Y, Z = np.meshgrid(np.arange(field_shape[0]), np.arange(field_shape[1]), np.arange(field_shape[2]), indexing='ij')

ax.quiver(X, Y, Z, VelRota["x"], VelRota["y"], VelRota["z"], length=0.1)
plt.show()

ax.quiver(X, Y, Z, VelLong["x"], VelLong["y"], VelLong["z"], length=0.1)
plt.show()
