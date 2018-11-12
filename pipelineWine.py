#!/usr/bin/python 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import time 

from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


startT = time.time()


winedf = pd.read_csv('winequality-red.csv',sep=';')

#print winedf.isnull().sum() check for missing data

print winedf.head(3)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ check whether the labels are unblanced or not
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

#print winedf.shape
#ylab = winedf[['quality']]
#print ylab.shape
#print winedf['quality'].value_counts() # indeed it is

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# check the correlation plot

#winecorr = winedf.corr()
#s=sns.heatmap(winecorr)
#s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
#s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)

#plt.show() # as expected high correlation between acidity and pH

# individual correlation plot
#plt.subplot(1,2,1)
#plt.scatter(winedf['fixed acidity'], winedf['pH'], s=winedf['quality']*5, color='magenta', alpha=0.3)
#plt.xlabel('Fixed Acidity')
#plt.ylabel('pH')
#plt.subplot(1,2,2)
#plt.scatter(winedf['fixed acidity'], winedf['residual sugar'], s=winedf['quality']*5, color='purple', alpha=0.3)
#plt.xlabel('Fixed Acidity')
#plt.ylabel('Residual Sugar')
#plt.tight_layout()
#plt.show()

X=winedf.drop(['quality'],axis=1)
Y=winedf['quality']

print type(X), type(Y)
print X.head(3)

#++++++++++++++++++++++++++++++++
# create the pipeline object
#++++++++++++++++++++++++++++++++
steps = [('scaler', StandardScaler()), ('SVM', SVC())]
pipeline = Pipeline(steps)


#++++++++++++++++++++++++++++++++++++++
#+ create the hyperparameter space
#++++++++++++++++++++++++++++++++++++++

parameteres = {'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}

#++++++++++++++++++++++++++++++++++++
#+ create train and test sets
#++++++++++++++++++++++++++++++++++++

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30, stratify=Y)

#print X_test.shape

#++++++++++++++++++++++++++++++
#+ Grid Search Cross Validation
#++++++++++++++++++++++++++++++
grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5)

grid.fit(X_train, y_train)

print "score = %3.2f" %(grid.score(X_test,y_test))

#pparam=pprint.PrettyPrinter(indent=2)

print grid.best_params_
endT = time.time()

print "total time elapsed = %3.3f"%(endT-startT)

