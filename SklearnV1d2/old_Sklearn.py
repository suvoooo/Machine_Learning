import time, psutil
import matplotlib.pyplot as plt
import sklearn
print ('sklearn version: ', sklearn.__version__) # 0.24

from sklearn.datasets import load_wine
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer

########################
X, y = load_wine(as_frame=True, return_X_y=True) # available from version >=0.23
########################

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, 
                                                    random_state=0)
X_train.head(3)


### standadrdizing transformation that leads to numpy array (dataframe in newer release)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #.set_output(transform='pandas') #change here, doesn't exist in version below 1.2
# AttributeError: 'StandardScaler' object has no attribute 'set_output'

scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)
print ('type after scaling: ', type(X_test_scaled))
# X_test_scaled.head(3) # throws error


###fetch openml (new addition in ver 1.2 is parser='pandas', memory efficient)
start_t = time.time()
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True,) # parser="pandas") # parser as a keyword in the 1.2 version
X = X.select_dtypes(["number", "category"]).drop(columns=["body"])
print ('check types: ', type(X), '\n',  X.head(3))
print ('check shapes: ', X.shape)
end_t = time.time()
print ('time taken: ', end_t-start_t)

process_names = [proc.name() for proc in psutil.process_iter()]
print (process_names)
print('cpu percent: ', psutil.cpu_percent())

#########################################
###### for simplicity ignore the nans
##########################################
print ('check for nans in columns: ', '\n', X.isna().sum())
# drop all the nans for making it suitable forGradientBoostingRegressor
X_NotNan =  X.dropna(how='any', inplace=False)
print ('check shapes after dropping nans: ', X_NotNan.shape)

nonan_indices = X_NotNan.index.to_list()
y_NotNan = y[y.index.isin(nonan_indices)]
print ('check shape y: ', y_NotNan.shape)
print ('check for indices: ', X_nonan.index.to_list())


#### pipeline for encoder + gradient boosting
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor #HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer

categorical_features = ["pclass", "sex", "embarked"]
model = make_pipeline(ColumnTransformer(transformers=[("cat", OrdinalEncoder(), categorical_features)], 
                                        remainder="passthrough",), 
                      GradientBoostingRegressor(random_state=0),).fit(X_NotNan, y_NotNan)

# gradientboosting doesn't work with nan entries


##########################
# partial dependence
##########################

from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import plot_partial_dependence

#fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
#disp = PartialDependenceDisplay.from_estimator(model, 
#                                               X_NotNan, features=["age", "sex", ("pclass", "sex")], 
#                                               categorical_features=categorical_features, ax=ax,)
# from_estimator method is non existent in older versions, but what about categorical features
#fig.savefig('./part_disp_old.png', dpi=200)

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('GradientBoostingRegressor')
GBR_disp = plot_partial_dependence(model, X_NotNan, ['age', 'fare', ('age', 'fare')], ax=ax)
fig.savefig('./part_disp_old_NotCat.png', dpi=200)

#fig, ax = plt.subplots(figsize=(12, 6))
#ax.set_title('GradientBoostingRegressor')
#GBR_disp = plot_partial_dependence(model, X_NotNan, ['age', 'sex', ('age', 'sex')], ax=ax)
#fig.savefig('./part_disp_old_wCat.png', dpi=200)
# valueerror
