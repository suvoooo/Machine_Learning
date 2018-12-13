#!/usr/bin/python

# this is checking and learning text classification using an amazing post by Susan Li, in Towards Data Science 

# the problem here is to assign a category when a new complaint comes in

import math 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np



from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2  
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
#from sklearn.model
 





complain_df = pd.read_csv('Consumer_Complaints.csv')

#print complain_df.head(3) 

# necessary columns required for text classifications are product (which will be label) and consumer complaint narrative (which will be feature(s))
#print complain_df.columns.values
df_columns = complain_df.columns.values
#print df_columns
#print type(df_columns)
df_columns[1] = 'product_label'

complain_df.columns = df_columns # 'product' has been changed to 'product_label'

#print complain_df.columns.values


#Product_df=complain_df[['Consumer complaint narrative']]
#print Product_df.head(5)

#++++++++++++++++++++++++++++++++++++++++++++++
# input: consumer_complaint_narrative 
#
# example: "I have outdated information on my credit
# report that I have previously disputed that...." 
#
# output: product
# example: Credit reporting
#++++++++++++++++++++++++++++++++++++++++++++++


columns = ['product_label', 'Consumer complaint narrative']

# we choose a new data-frame with only these columns as 'Consumer complaint narrative'
# represents feature and 'product_label' is indeed what we want to predict i.e. label

new_df = complain_df[columns]
#print new_df.head(3)
new_df_columns = new_df.columns.values
#print new_df_columns
new_df_columns[1]='consumer_complaint_narrative'
new_df.columns = new_df_columns
#print new_df.shape # (1144848,2)
#print new_df['product_label'].unique() # check the unique product labels
# drop rows that have NaN values in Consumer complaint narrative

new_df = new_df.dropna(axis=0, how='any') # # drop the rows which contains missing values; any NA 
#print new_df.shape #(332361, 2)

# create a new column where product_label is catagorized. 

new_df['category'] = new_df['product_label'].factorize()[0] # to quote scikit_learn 'factorize is useful for obtaining numeric representation
# of an array when all that matters is identifying missing values. available as Series.factorize()'. new_df['product_label'] is the series here.

#print new_df.head(4)



category_id_df=new_df[['product_label','category']].drop_duplicates().sort_values('category') # drop duplicates which matches column product_label and category and then sort according to category, there will
# be use of this for later purpose


#print category_id_df.shape # (18,2)
#print category_id_df.head(18)


category_to_id=dict(category_id_df.values)
#print category_to_id


#for k, d in sorted(category_to_id.iteritems()): # we will use this later
#	print k, 'correspond to', d # sweeeeet 
	



#__________________________________________________________________
#+ plot to see that product label is biased towards  
#  credit complains
#__________________________________________________________________
#fig = plt.figure(figsize=(11.,10.))
#fig = plt.figure()
#fig.patch.set_facecolor('white')

#new_df.groupby('product_label').consumer_complaint_narrative.count().plot.bar(ylim=0, rot=75, fontsize=7)
#plt.show() # the plot shows that few of the product labels totally dominate the number of complaints and we need to avoid our model from
# being biased towards the majority of classes. It could be problem for handling data-sets of fraud detection or cancer prediction but here
# it helps since the classifier may give high prediction accuracy for majority of the labels. 
#________________________________________________________________________________
#






# +++++++++++++++++++++++++++++++++++++++++++++++++
#+ How to represent text
# +++++++++++++++++++++++++++++++++++++++++++++++++
print "shape of new_df", new_df.shape
print new_df.head(4)
new_df=new_df[100:25000] # we select a smaller data-set otherwise tfidf method will cause segmentation fault (memory error)
new_df.to_csv("Consumer_Complaints_short.csv", sep='\t', encoding='utf-8')
print "after selecting few rows", new_df.shape
print len(new_df['category'].unique()) # check that even selecting a smaller sample won't reduce the unique category 
# learning algorithms and classifiers can not directly process text in original form, as most them are dependent on 
# numerical feature vector with fixed size rather than text of variable length. so the texts need to be converted 
# into something more manageable representation. 

# usual method is to use bag of words model; where occurences of words are checked but orderings are ignored. 

# we will use tfidfvectorizer which converts a collection of raw documents to a matrix of tf-idf features.   


# sublinear_df is set to True to use a logarithmic form for frequency.
# min_df is the minimum numbers of documents a word must be present in to be kept.
# norm is set to l2, to ensure all our feature vectors have a euclidian norm of 1.
# stop_words remove "a", "the" from the files (here the consumer complain).


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1,2), stop_words = 'english')

 


features_text = tfidf.fit_transform(new_df.consumer_complaint_narrative).toarray()
print features_text.shape # so here we see 24900 complains which are represented by 63530 features
labels = new_df.category




#for Product, category in sorted(category_to_id.items()):
#	features_chi2=chi2(features_text, labels==category)
#	indices = np.argsort(features_chi2[0])
#	feature_names = np.array(tfidf.get_feature_names())[indices]
#	unigrams = [v for v in feature_names if len(v.split(' '))==1]
#	bigrams = [v for v in feature_names if len(v.split(' '))==2]
#	print "# '{}': ".format(Product)
#	print "Most correlated unigrams:\n. {}".format('\n.'.join(unigrams[-2:]))
#	print "Most correlated bigrams: \n. {}".format('\n.'.join(bigrams[-2:]))






X_train, X_test, y_train, y_test = train_test_split(new_df['consumer_complaint_narrative'], new_df['product_label'], test_size=0.3, random_state=30)

count_vect = CountVectorizer()
print "train and test length", len(X_train), len(X_test)



X_train_count = count_vect.fit_transform(X_train)
#print "train_count", len(X_train_count)


X_test_count = count_vect.transform(X_test) # do not apply fit method on test data. Only transform it to a matrix of token counts using CountVectorizer

tfidf_transform = TfidfTransformer()

X_train_tfidf = tfidf_transform.fit_transform(X_train_count)
X_test_tfidf = tfidf_transform.fit_transform(X_test_count)


clf = MultinomialNB().fit(X_train_tfidf, y_train)


#++++++++++++++++++++++++++++++++++++
#+ perdiction time
#++++++++++++++++++++++++++++++++++++
print clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."]))

print(clf.predict(count_vect.transform(["I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"])))









#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ Final part : Selecting which algorithm works best
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# create a list of models 

test_ML_Models = [MultinomialNB(), LinearSVC(), LogisticRegression(), RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)] #n_estimator: number of trees in forest, max_depth=

test_ML_Models_columns = [] 

test_ML_df = pd.DataFrame(columns=test_ML_Models_columns)

print test_ML_df.head(3)


row_index = 0

for algorithm in test_ML_Models:
	predicted = algorithm.fit(X_train_tfidf, y_train)#.predict(X_test)
	test_ML_Models_name = algorithm.__class__.__name__
	test_ML_df.loc[row_index,'test_ML_Models_name'] = test_ML_Models_name
	test_ML_df.loc[row_index, 'Train Accuracy'] = round(algorithm.score(X_train_tfidf,y_train),3)
	test_ML_df.loc[row_index, 'Test Accuracy']	= round(algorithm.score(X_test_tfidf,y_test),3)
	row_index = row_index + 1

	 
test_ML_df.sort_values(by=['Train Accuracy'], ascending=False, inplace=True)

print test_ML_df.head(4)# support vector machine has the highest accuracy on train (93%) and test (71%) data.




