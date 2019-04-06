#!/usr/bin/python

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

movies_df = pd.read_csv("movie_metadata.csv")
#print movies_df.head(3)
#print movies_df.describe()
# check the column names
#print "column names: ", movies_df.columns.values
#print movies_df.dtypes
#print "data-frame shape: ", movies_df.shape # (5043, 28)

# check null data 
#print "null values: \n", 
#print movies_df.isna() # same as movies_df.isnull().sum()
#print movies_df.isna().sum()
#print "null values", movies_df.isnull().values.any() 
#print "total null values", movies_df.isna().sum().sum()
#print movies_df.describe()

#clean_movies_df = movies_df.dropna(how='any')
movies_df = movies_df.dropna(how='any')

#print "new dataframe shape: ", clean_movies_df.shape # (3756, 28)
#print "old dataframe shape: ", movies_df.shape
#use fillna attribute of pandas 
#for filling up missing values in all columns  
#movies_df.fillna(value=0, inplace=True)

# for some specfific columns we can just choose select those columns 
# movies_df[['gross', 'budget']]=movies_df[['gross', 'budget']].fillna(value=0)
# filling with the mean
#movies_df['budget'].fillna(movies_df[budget].mean(), inplace=True) 

#for this dataframe we can use 'missing' in the columns of object data types for example language or movie_imdb_link

#movies_df['language'].fillna("no info", inplace=True) 

#print "null values", movies_df.isna().sum() # now no missing values in languages column!

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# check if there are duplicate rows or not 

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#duplicate_rows_df = movies_df[movies_df.duplicated()]
#print "number of duplicate rows: ", duplicate_rows_df.shape


#print duplicate_rows_df.head(6)
#print movies_df['movie_imdb_link'].head(3)

#duplicate_rows_df_imdb_link = movies_df[movies_df.duplicated(['movie_imdb_link'])]
#print duplicate_rows_df_imdb_link.head(3)
#print "shape of duplicate dataframe with same imdb link", duplicate_rows_df_imdb_link.shape

#print len(movies_df.movie_imdb_link.unique())
# select duplicate rows except first occurences, consider all columns  
#duplicate_rows_df = movies_df[movies_df.duplicated()]
#print type(duplicate_rows_df) # dataframe 
#print duplicate_rows_df.shape


#++++++++++++++++++++++++++++++++
# drop_duplicates
#++++++++++++++++++++++++++++++++

#Drop duplicate rows (duplicate values for all entries)
#print "shape of dataframe before dropping duplicates", movies_df.shape
#0print "shape of dataframe after dropping duplicates", movies_df.drop_duplicates().shape


#+++++++++++++++++++++++++++++++++++++++++++++++
#+ discretization or binning
#+++++++++++++++++++++++++++++++++++++++++++++++

#print movies_df['imdb_score'][5:10]
# check the miminmum value of all columns
#print "minimum values of all cloumn:", 
#print '\n'
#print movies_df.min()

#print movies_df['imdb_score'].idxmax()
#print movies_df.loc[movies_df['imdb_score'].idxmax(), 'movie_title']
#print movies_df['movie_title'].loc[2764:2767]
#print movies_df['budget'].idxmax()
#print movies_df[['movie_title','budget']].loc[2986:2990]

# check the distribution of imdb score
#fig = plt.figure(figsize=(10,7))
#sns.distplot(movies_df['imdb_score'])
#plt.xlabel('IMDB Score', fontsize=12)
#sns.jointplot(x='budget', y='imdb_score', data=movies_df); 
#plt.show()


# based on the 'imdb_score' we will discritize the movies in 3 categories ['shyte', 'moderate', 'good']
# similar with pd.cut method described in McKinney's book
#op_labels = ['shyttte', 'moderate', 'good']
#category = [0.,4.,7.,10.]
#movies_df['imdb_labels'] = pd.cut(movies_df['imdb_score'], labels=op_labels, bins=category, include_lowest=False)
#print movies_df[['movie_title', 'imdb_score', 'imdb_labels']][209:220]



#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#_ Removing outliers from the dataframe 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#___________________________________________
# First check with box plot 
#___________________________________________

# let's try to plot boxplot with seaborn 
#sns.boxplot(x=movies_df['facenumber_in_poster'], color='lime')
#plt.xlabel('No. of Actors Featured in Poster', fontsize=14)
#plt.show()

#print "min no", movies_df['facenumber_in_poster'].min()
#print "max no: index", movies_df['facenumber_in_poster'].idxmax()
#print movies_df[['movie_title', 'facenumber_in_poster']].iloc[movies_df['facenumber_in_poster'].idxmax()]
#print movies_df['facenumber_in_poster'].describe()
# budget column is massively varying. so we are in dire need to drop outliers
#best option for dropping outliers is to use zscore method and reject all rows in non-object type columns 


#____________________________________________________________________________________
# Use Z Score from Scipy Stats 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html
#____________________________________________________________________________________


# will try first using the z score method 
# z score is basically how many standard deviation away a point is from the mean. Closer the value of z is towards 0, the data point is closer to the mean. 

#first detect outlier for a particular column


#print "data types: \n", movies_df.dtypes
#print "shape before :", movies_df.shape

movies_df_num = movies_df.select_dtypes(exclude=['object'])
#print "shape after excluding object columns: ", movies_df_num.shape 
# 12 object type columns were dropped. 

#use z score for all columns in the new data frame 
movies_df_Zscore = movies_df_num[(np.abs(stats.zscore(movies_df_num))<=3).all(axis=1)]
#print "shape after rejecting outliers: ", movies_df_Zscore.shape
movies_df_Zscore_usr_rev = movies_df_num[(np.abs(stats.zscore(movies_df_num[['num_user_for_reviews']]))<=3).all(axis=1)]
print type(movies_df_Zscore_usr_rev)

#fig = plt.figure(figsize=(12,8))
#plt.subplot(1,2,1)
#sns.boxplot(x=movies_df['num_user_for_reviews'], color='lime')
#plt.xlabel('No. of Actors Featured in Poster (After Using Z Score)', fontsize=14)
#plt.subplot(1,2,2)
#sns.distplot(movies_df_Zscore['num_user_for_reviews'], color='lime')
#plt.tight_layout()
#plt.show()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ use numpy 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

movies_num_usr_rev= movies_df[np.abs(movies_df.num_user_for_reviews-movies_df.num_user_for_reviews.mean()) <= (3*movies_df.num_user_for_reviews.std())]
print type(movies_num_usr_rev)
print movies_num_usr_rev.head(3)

fig = plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.distplot(movies_num_usr_rev['num_user_for_reviews'], color='lime')
plt.xlabel('No. of Users to Review (Numpy only on "num_user_for_reviews")', fontsize=13)
plt.subplot(1,2,2)
sns.distplot(movies_df_Zscore_usr_rev['num_user_for_reviews'], color='lime')
plt.xlabel('No. of Users to Review (Scipy Z Score on "num_user_for_reviews" Column)', fontsize=13)
plt.tight_layout()
plt.show()


#=============================================
#=      Some more checks
#=============================================

#movies_df['budget_zscore'] = movies_df['budget'].stats
 
#print type(stats.zscore(movies_df['budget'])) # numpy array 
#budg_zsc = stats.zscore(movies_df['budget'])
#print type(budg_zsc)
#print np.where(budg_zsc > 3.0)[0]
#print budg_zsc[3259]
#print budg_zsc[100:133] # only nan ? because the nan values aren't dropped yet
# In that case we use mean and std instance of dataframe 
#print type( (movies_df.budget - movies_df.budget.mean())/movies_df.budget.std(ddof=0))
#mov_df_budget_zscore = (movies_df.budget - movies_df.budget.mean())/movies_df.budget.std(ddof=0)
#print mov_df_budget_zscore[100:131]
#mov_budg_zsc_arr =  mov_df_budget_zscore.values
#print mov_budg_zsc_arr[50:55]
#print np.where(mov_budg_zsc_arr > .60)[0] 		 



 


