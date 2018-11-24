#!/usr/bin/python 

import math 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


matplotlib.rcParams.update({'font.size': 12})

circle1 = plt.Circle((0,0),1, color='magenta', alpha=0.4)

fig,ax= plt.subplots() 

ax.add_artist(circle1)


ax.set_xlim(-5.2,5.2)
ax.set_ylim(-5.2,5.2)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)


xp = np.linspace(-10,10,num=1000)

print xp


kval = [-3,-2,-1,0,1,2,4,8.125,11]


yp0 = [(8*(x**2) - kval[0])/2.0 for x in xp]
yp1 = [(8*(x**2) - kval[1])/2.0 for x in xp]
yp2 = [(8*(x**2) - kval[2])/2.0 for x in xp]
yp3 = [(8*(x**2) - kval[3])/2.0 for x in xp]
yp4 = [(8*(x**2) - kval[4])/2.0 for x in xp]
yp5 = [(8*(x**2) - kval[5])/2.0 for x in xp]
yp6 = [(8*(x**2) - kval[6])/2.0 for x in xp]
yp7 = [(8*(x**2) - kval[7])/2.0 for x in xp]
yp8 = [(8*(x**2) - kval[8])/2.0 for x in xp]





#print yp


plt.plot(xp, yp0,color='hotpink',label=r'$f(x,y)=-3.0$')
plt.plot(xp, yp1,color='red',linestyle='-.',label=r'$f(x,y)=-2.0$')
plt.plot(xp, yp2,color='olive',label=r'$f(x,y)=-1.0$')
plt.plot(xp, yp3,color='lime',label=r'$f(x,y)=0.0$')
#plt.plot(xp, yp4,color='aqua',label=r'$f(x,y)=1.0$')
plt.plot(xp, yp5,color='aqua',label=r'$f(x,y)=2.0$')
#plt.plot(xp, yp5,color='deepskyblue',label=r'$f(x,y)=4.0$')
plt.plot(xp, yp7,color='navy',linestyle='--',label=r'$f(x,y)=8.125$')
plt.plot(xp, yp8,color='gray',label=r'$f(x,y)=11.0$')

plt.axhline(y=0, linestyle=':', linewidth=1, color='orange')
plt.axhline(y=1, linestyle=':', linewidth=1, color='orange')
plt.axhline(y=-1, linestyle=':', linewidth=1, color='orange')


plt.axvline(x=0, linestyle=':', linewidth=1, color='orange')
plt.axvline(x=1, linestyle=':', linewidth=1, color='orange')
plt.axvline(x=-1, linestyle=':', linewidth=1, color='orange')



plt.annotate('Minima',color='red',xy=(-0.49, 2.34), xytext=(-3.0,2.24), ha='center', arrowprops=dict(facecolor='red', edgecolor='red',shrink=0.05,  width=1))
plt.annotate('Maxima',color='navy',xy=(0.67, -2.), xytext=(3.0,-2.05), ha='center', arrowprops=dict(facecolor='navy', edgecolor='navy',shrink=0.05,  width=1))

#print  Ypara1


plt.title ('Example of Solving Equation with Constraint; Lagrange Multiplier')
plt.legend(fontsize=12)
plt.show()
