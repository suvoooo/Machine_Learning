#!/usr/bin/python

import math 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(10)

#definition of probability density function 
# pdf is for a continuous random variable, is a function whose value at any given sample (or data point) in the sample space can be interpreted as providing a relative likelihood that the value of the random variable would equal that sample space. 
   
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 10, 0.001)
Xarr = np.random.normal(loc=0.0, scale=5.0, size=1000)
print np.mean(Xarr), np.std(Xarr)
sns.distplot(Xarr, color='lime', hist_kws=dict(alpha=0.3))



pl_1_std = np.mean(Xarr) + np.std(Xarr)
mi_1_std = np.mean(Xarr) - np.std(Xarr)

pl_3_std = np.mean(Xarr) + (3*np.std(Xarr))
mi_3_std = np.mean(Xarr) - (3*np.std(Xarr))
#+++++++++++++++++++++++++++++++++++++++++++++++++
# Mean = 0, SD = 2.

#plt.plot(x_axis, norm.pdf(x_axis,0,5), color='purple', linestyle='--')
#plt.plot(x_axis, norm.pdf(x_axis,0,2), color='magenta', linestyle='-.')
#+++++++++++++++++++++++++++++++++++++++++++++++++

plt.axvline(pl_1_std, ymin=0, ymax = 0.53,linewidth=4, color='orange')
plt.axvline(pl_3_std, linestyle='-.', linewidth=4, color='magenta', ymin=0, ymax=0.05)

plt.axvline(mi_1_std, label=r'$-1\, \sigma$', ymin=0, ymax = 0.53, linewidth=4, color='orange')
plt.axvline(mi_3_std, linestyle='-.', color='magenta', label=r'$-3\, \sigma$', linewidth=4, ymin=0, ymax=0.05)

#plt.legend(fontsize=14)
plt.text(-16.3, 0.02, r'$-3\, \sigma$', fontsize=13, rotation='vertical', color='magenta',
		 bbox=dict(facecolor='none', edgecolor='lavender', boxstyle='round,pad=0.7'))


plt.text(15.5, 0.02, r'$+ 3\, \sigma$', fontsize=13, rotation='vertical', color='magenta',
		 bbox=dict(facecolor='none', edgecolor='lavender', boxstyle='round,pad=0.7'))

plt.text(-6.5, 0.07, r'$-1\, \sigma$', fontsize=13, rotation='vertical', color='orange',
		 bbox=dict(facecolor='none', edgecolor='azure', boxstyle='round,pad=0.7'))


plt.text(6.5, 0.07, r'$+ 1\, \sigma$', fontsize=13, rotation='vertical', color='orange',
		 bbox=dict(facecolor='none', edgecolor='azure', boxstyle='round,pad=0.7'))


plt.show()
