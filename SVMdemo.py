#!/usr/bin/python 

import numpy as np 
import math
import sklearn 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.datasets.samples_generator import make_circles

X,y = make_circles(90, factor=0.2, noise=0.1)
#print type(X)


#plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap='seismic')

r = np.exp(-(X**2).sum(1))
zaxis = [0.2,0.4,0.6,0.8, 1.0]
zaxislabel = [r'0.2',r'0.4', r'0.6', r'0.8', r'1.0']

fig = plt.figure()
ax = Axes3D(fig)


def plot3dim():
	#ax=plt.subplot(111, projection='3d')
	ax.scatter(X[:,0], X[:,1], r, c=y, s=50, cmap='seismic')
	#ax.view_init(elev=elev,azim=azim)
	ax.set_xlabel('X')
	ax.set_ylabel('y')
	ax.set_zlabel('!! SHAKE !!', fontsize=15, labelpad=-1, color='lime')
	ax.set_zticklabels(zaxislabel, fontsize=7)
	ax.set_zticks(zaxis)
	ax.grid('False')
	return fig, 

def animate(k):
	ax.view_init(elev=k,azim=30)
    #return fig, 

ani = animation.FuncAnimation(fig, animate, init_func=plot3dim, frames=360, interval=30, blit=False)

#ani.save('SVManim.mp4', fps=30, dpi=200, extra_args=['-vcodec', 'libx264'])

#plot3dim(elev=10, azim=-20, X=X,  y=y)

plt.show()
