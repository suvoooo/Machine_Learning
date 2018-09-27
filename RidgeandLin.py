#!/usr/bin/python 

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12})


from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

boston=load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)
#print boston_df.info()

X=boston_df
#print type(X) # X is in pandas data-frame format 
#print X[0:10] check 
Y=boston.target
#print Y[0:10]
#print "target variable type:", type(Y)

# add another column that contains the house prices which in scikit learn datasets are considered as 
boston_df['Price']=boston.target
#print boston_df.head(3)


newX = boston_df
#print newX[0:5]

newX=boston_df.drop('Price',axis=1)
print newX[0:3]
newY=boston_df['Price']

#print type(newY)# pandas core frame

X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)
print len(X_test), len(y_test)


lr = LinearRegression()
lr.fit(X_train, y_train)


rr = Ridge(alpha=0.01) # higher the alpha value, more restriction on the coefficients; low alpha more generalization, coefficients are barely
# restricted and in this case linear and ridge regression resembles
rr.fit(X_train, y_train)



rr100 = Ridge(alpha=100)
rr100.fit(X_train, y_train)

train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)


Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)


Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)




print "linear regression train score:", train_score
print "linear regression test score:", test_score
print "ridge regression train score low alpha:", Ridge_train_score
print "ridge regression test score low alpha:", Ridge_test_score
print "ridge regression train score high alpha:", Ridge_train_score100
print "ridge regression test score high alpha:", Ridge_test_score100




# for some reason the test score are higher than the training scores, possibly because of low sample numbers, this can be checked by changing the test_size fraction from 0.2 to 0.3 where test_score eventually falls below the training score 


plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # alpha here is for transparency
plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency

plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()
