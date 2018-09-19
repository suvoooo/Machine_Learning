#!/usr/bin/python 

# this program check the effect of pca (dimensionality reduction) on cancer data 


# check the histograms at the beginning.  Cancer Data set has 30 features 

from sklearn.datasets import load_breast_cancer
from sklearn.cross_validation import train_test_split
from matplotlib.pyplot import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import pandas as pd

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ effect of features on target
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cancer=load_breast_cancer()
cancer_df=pd.DataFrame(cancer.data,columns=cancer.feature_names)

#print cancer_df[feature_mean].corr()


print cancer_df.info()
#print cancer_df.head(4)

feature_mean=list(cancer_df.columns[0:10])
feature_worst=list(cancer_df.columns[20:31])

#print len(cancer.target)

#print cancer.DESCR # it's better to know about the data-set 
#print len(cancer.data[cancer.target==1]) # it's confusing because benign is listed as 1 




#print cancer.feature_names

#plt.subplot(1,2,1)
#plt.scatter(cancer_df['worst symmetry'],cancer_df['worst texture'],s=cancer_df['worst area']*0.05,color='magenta',label='check',alpha=0.3)
#plt.xlabel('Worst Symmetry',fontsize=12)
#plt.ylabel('Worst Texture',fontsize=12)
#plt.subplot(1,2,2)
#plt.scatter(cancer_df['mean radius'],cancer_df['mean concave points'],s=cancer_df['mean area']*0.05,color='purple',label='check',alpha=0.3)
#plt.xlabel('Mean Radius',fontsize=12)
#plt.ylabel('Mean Concave Points',fontsize=12)
#plt.tight_layout()
#plt.show()

'''

fig,axes =plt.subplots(10,3, figsize=(12, 9))

malignant=cancer.data[cancer.target==0]
benign=cancer.data[cancer.target==1]



ax=axes.ravel()

for i in range(30):
	_,bins=np.histogram(cancer.data[:,i],bins=40)
	ax[i].hist(malignant[:,i],bins=bins,color='r',alpha=.5)
	ax[i].hist(benign[:,i],bins=bins,color='g',alpha=0.3)
	ax[i].set_title(cancer.feature_names[i],fontsize=9)
	ax[i].axes.get_xaxis().set_visible(False)	
	ax[i].set_yticks(())

ax[0].legend(['malignant','benign'],loc='best',fontsize=8)
plt.tight_layout()
#fig.subplots_adjust(lspace=2)
#plt.suptitle("Cancer Histograms", fontsize=4)		

plt.show() # not given in Muller book but it's necessary to see the plots 

# from the plots 1>worst smoothness and 2> worst perimeter produces well separated histograms # important features to distinguish between 
# malignant and benign tumors

# before aplying pca it's necessary to process the features so that they lie within similar range (StandardScaler or MinMaxScaler)

'''
#+++++++++++++++++++++++++++++++
#+ before PCA scale the data
#+++++++++++++++++++++++++++++++

scaler = StandardScaler() # standardized feature by removing mean and scaled to unit variance 
scaler.fit(cancer.data)

X_scaled = scaler.transform(cancer.data)


print "after scaling", X_scaled.min(axis=0)



#+++++++++++++++++++++++++++++++++++++++++++
#+ PCA (Principal component analysis)
#+++++++++++++++++++++++++++++++++++++++++++
pca = PCA(n_components=3) # instantiate the PCA and keep the first n components
pca.fit(X_scaled)
# now transform 

x_pca=pca.transform(X_scaled)

# check the shape 

print x_pca.shape
#print pca.explained_variance

ex_variance=np.var(x_pca,axis=0)

ex_variance_rat=ex_variance/np.sum(ex_variance)

print ex_variance_rat

'''
#+++++++++++++++++++++++++++++
#+ plot the pcs
#+++++++++++++++++++++++++++++

#fig=plt.figure(figsize=(6,4))


#p=plt.scatter(x_pca[:,2],x_pca[:,0],s=40,c=cancer.target) # c =cancer.target tells that minimum/maximum values of c corresponds to bottom/up of the plots 



Xax=x_pca[:,0]
Yax=x_pca[:,1]
labels=cancer.target
#labels=['Malignant','Benign']
cdict={0:'red',1:'green'}
labl={0:'Malignant',1:'Benign'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5}
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
	ix=np.where(labels==l)
	ax.scatter(Xax[ix],Yax[ix],c=cdict[l],label=labl[l],s=40,marker=marker[l],alpha=alpha[l])




plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)

plt.legend()


plt.show() # Malignant data are way more spreadout than benign data




#print pca.components_.shape

'''

#+++++++++++++++++++++++++++++++++++++++
#+ visualize the effect of PCA
#+++++++++++++++++++++++++++++++++++++++


#plt.matshow(pca.components_,cmap='viridis')
#plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
#plt.colorbar()
#plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=65,ha='left')
#plt.tight_layout()
#plt.show()# since the direction of the arrow doesn't matter in pca plot, it can be concluded that the features have a strong correlation 



#+++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ finally check the correlation of the features
#+++++++++++++++++++++++++++++++++++++++++++++++++++++

#correlation = cancer.feature_.corr()

s=sns.heatmap(cancer_df[feature_worst].corr(),cmap='coolwarm') # fantastic tool to study the features 
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
plt.show()# super happy to complete this project


# finished
