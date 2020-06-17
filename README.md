# Machine_Learning
# This repository contains some of the most important machine learning and data-analysis techniques.
## When the new files will be added, corresponding description will also be added with file name and DDMMYY. 

*PCA_Muller.py 190818:* Principal component analysis example with breast cancer data-set. Detailed description of this code is discussed in [Towards Data Science](https://towardsdatascience.com/dive-into-pca-principal-component-analysis-with-python-43ded13ead21). Base of this is taken from Prof. A. Muller's [Machine Learning](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413) book.    

*270918: RidgeandLin.py, LassoandLin.py:* Lasso and Ridge regression examples: The concepts and discussion of the results are described [here](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b). Base of this is taken from Prof. A. Muller's [Machine Learning](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413) book.     

*081018: bank.csv*, data set of selling products of a portuguese company to random customers over phone call(s). Detailed description are available [here](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

*161018: gender_purchase.csv*, data-set of two columns describing customers buying a product depending on gender.

*111118: winequality-red.csv*, red wine data set, where the output is the quality column which ranges from 0 to 10.

*121118: pipelineWine.py*, Contains a simple example of applying pipeline and gridsearchCV together using the red wine data. More description can be found [here](https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976). 

*24112018: lagmult.py*, this program just demonstrate a simple constrained optimization problem using figures. Uses Lagrange Multiplier method.  

*11122018: Consumer_Complaints_short.csv*, 3 columns describing the complaints, product_label and category. Complete file can be obtained from [Govt.data](https://catalog.data.gov/dataset/consumer-complaint-database/resource/2f297213-7198-4be1-af1e-2d2623e7f6e9). File size is around 650 MB. More details about the usage of this file will be uploaded soon when the text classification program is ready. 

*13122018: Text-classification_compain_suvo.py*, Classify the consumer complaints data, which is already described above. The file deals with the complete data-set (650 MB). After testing several ML algorithms, Linear SVM works best. Higher the computer resources, higher amount of rows can be considered for TfidfVectorizer. 

*1912018: SVMdemo.py*, this program shows the effect of using RBF kernel to map from 2d space to 3d space. Animation requires ffmpeg in unix system. 

*05032019: IBM_Python_Web_Scrapping.ipynb*, Deals with basic web scrapping, string handling, image manipulation while we generate fake cover for our band.

*06042019: datacleaning*, Folder containing files and images related to data cleaning with pandas. For more details check [Medium](https://medium.com/@saptashwa/data-handling-using-pandas-cleaning-and-processing-3aa657dc9418). 

*09062019: DBSCAN_Complete*, Folder containing files and images related to application of DBSCAN algorithm to cluster Weather Stations in Canada. Apart from ususal Scikit-learn, numpy, pandas, I have used Basemap to show the clusters on a map. More details can be found in [Medium](https://medium.com/@saptashwa/dbscan-algorithm-complete-guide-and-application-with-python-scikit-learn-d690cbae4c5d).

*13072019: SVM_Decision_Boundary*, I set up a pipeline with StandardScaler, PCA, SVM and, performed grid-search cross-validation to find best-fit parameters, using which the decision function contours of SVM classifier for binary classification are plotted. Read in [TDS](https://towardsdatascience.com/visualizing-support-vector-machine-decision-boundary-69e7591dacea) for more.     

*28122019: DecsTree*, Folder contains notebook using a decision tree classifier on the [Bank Marketing Data-Set](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing). Best parameters are obtained using Grid Search Cross Validation. Also materials used for the [TDS post](https://towardsdatascience.com/understanding-decision-tree-classification-with-scikit-learn-2ddf272731bd) are included.  

*07032020: Conjugate Prior*, Folder contains a notebook where concept of conjugate prior is discussed including an introduction to [PyMC3](https://docs.pymc.io/). More details can be found [here](https://towardsdatascience.com/understanding-conjugate-priors-21b2824cddae).  

*29052020: ExMax_Algo*, Folder contains a notebook completely explaining the Expectation Maximization algorithm. Implemented with a reference to Gaussian Mixture Models. More details in [TDS](https://towardsdatascience.com/latent-variables-expectation-maximization-algorithm-fb15c4e0f32c).  
