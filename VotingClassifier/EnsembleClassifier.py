##########################
# import Libraries
#########################

from itertools import combinations

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV



####################
# Load Data
# Original (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
###################

heart_df = pd.read_csv('./heart.csv', )
# heart_df.head(4)

###########
print ('check label counts: ', heart_df['HeartDisease'].value_counts())

label_class = {0:'Healthy', 1:'Ill'}
heart_df['ClassLabel'] = heart_df['HeartDisease'].map(label_class)
print (heart_df.head(5))

#####################################
# Plots for Checking Some Features
#####################################

#countplot
fig, axs = plt.subplots(figsize=(8, 6))
sns.countplot(data = heart_df, x=heart_df['ClassLabel'], hue='Sex', ax=axs)
# plt.show

#violinplots
fig, axs = plt.subplots(1, 2, figsize=(9, 5))
axs=axs.flatten()
sns.violinplot(data = heart_df, x='Sex', y='Cholesterol',  
               hue='ClassLabel', palette='cubehelix',
               split=True, ax=axs[0])
sns.violinplot(data = heart_df, x='Sex', y='RestingBP',  
               hue='ClassLabel', palette='cubehelix',
               split=True, ax=axs[1])
plt.tight_layout()
# plt.show

#boxplots
### Check Few Parameter Distributions:

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
axes = axes.flatten()
sns.boxplot(x=heart_df['Age'], color='orange',  ax=axes[0])
sns.boxplot(x=heart_df['Cholesterol'], color='orange',  ax=axes[1])
sns.boxplot(x=heart_df['RestingBP'], color='orange',  ax=axes[2])
sns.boxplot(x=heart_df['MaxHR'], color='orange',  ax=axes[3])
plt.tight_layout()
# plt.show

########
# Dummies for categorical variables
########

selected_rows_heart = pd.get_dummies(heart_df, drop_first=True)
# selected_rows_heart.head(3)

###################
# Training Data Preparation
###################

X = selected_rows_heart.drop(['HeartDisease', 'ClassLabel_Ill'], axis=1)
y = selected_rows_heart['HeartDisease']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.25, random_state = 42)

print ('train data shape: ', X_train.shape, y_train.shape)
print ('test data shape: ', X_test.shape, y_test.shape)

### standardize the dataframe (training and test data should be separated)
### we only standardize the numerical columns

numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.fit_transform(X_test[numerical_cols])

# X_train.head(3)

#########################
# Check Model Performance Separately
#########################

# Initialize the classifiers
svc_classifier = SVC()
logreg_classifier = LogisticRegression()
adaboost_classifier = AdaBoostClassifier()

# Train the models on the training data
svc_classifier.fit(X_train, y_train)
logreg_classifier.fit(X_train, y_train)
adaboost_classifier.fit(X_train, y_train)

# Step 3: Evaluate the models
# Make predictions on the test data
svc_predictions = svc_classifier.predict(X_test)
logreg_predictions = logreg_classifier.predict(X_test)
adaboost_predictions = adaboost_classifier.predict(X_test)

# Evaluate the performance of each classifier using various metrics

svc_precision = precision_score(y_test, svc_predictions, average='weighted')
logreg_precision = precision_score(y_test, logreg_predictions, average='weighted')
adaboost_precision = precision_score(y_test, adaboost_predictions, average='weighted')

svc_recall = recall_score(y_test, svc_predictions, average='weighted')
logreg_recall = recall_score(y_test, logreg_predictions, average='weighted')
adaboost_recall = recall_score(y_test, adaboost_predictions, average='weighted')

svc_f1 = f1_score(y_test, svc_predictions, average='weighted')
logreg_f1 = f1_score(y_test, logreg_predictions, average='weighted')
adaboost_f1 = f1_score(y_test, adaboost_predictions, average='weighted')

# Print the evaluation metrics
print("SVC, LogReg, AdaBoost Precisions:", svc_precision, logreg_precision, adaboost_precision)
print("SVC, LogReg, AdaBoost Recalls:", svc_recall, logreg_recall, adaboost_recall)
print("SVC, LogReg, AdaBoost F1-scores:", svc_f1, logreg_f1, adaboost_f1)


#################
# Plot AdaBoost Feature Importances
#################

n_features = len(X_train.columns)

sns.set(style="whitegrid")

fig = plt.figure(figsize=(15, 11))
fig.tight_layout()
plt.bar(range(n_features), adaboost_classifier.feature_importances_, color="magenta", align="center", alpha=0.6)
plt.xticks(np.arange(n_features), X_train.columns.to_list(), rotation=80, fontsize=11)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Feature Importance", fontsize=14)
# plt.savefig("./Feature_Importance_AdaBoost.png", dpi=300, bbox_inches='tight')
# xticks are not clipped with 'bbox'
# plt.show

############################
# Check Performance with VC
############################



# Create the Voting Classifier
# One can choose either 'hard' voting or 'soft' voting.
# 'hard' voting - Majority vote (best classifier is choosen)
# 'soft' voting - Weighted average of probabilities (combine all; we can also specify the weight array;
# usually all models have weight 1)

svc_classifier = SVC(probability=True)
voting_classifier_hard = VotingClassifier(estimators=[('svm', svc_classifier), ('logreg', logreg_classifier), ('adaboost', adaboost_classifier)],
                                          voting='hard')

voting_classifier_soft = VotingClassifier(estimators=[('svm', svc_classifier), ('logreg', logreg_classifier), ('adaboost', adaboost_classifier)],
                                          voting='soft')

# Step 4: Train the Voting Classifier on the training data
voting_classifier_hard.fit(X_train, y_train)
voting_classifier_soft.fit(X_train, y_train)

# Step 5: Make predictions on the test data
voting_predictions_hard = voting_classifier_hard.predict(X_test)
voting_predictions_soft = voting_classifier_soft.predict(X_test)


vc_precision_soft = precision_score(y_test, voting_predictions_soft)
vc_precision_hard = precision_score(y_test, voting_predictions_hard,)

vc_recall_soft = recall_score(y_test, voting_predictions_soft)
vc_recall_hard = recall_score(y_test, voting_predictions_hard)

vc_f1_soft = f1_score(y_test, voting_predictions_soft)
vc_f1_hard = f1_score(y_test, voting_predictions_hard)


# Print the evaluation metrics
print("VC_soft, VC_hard Precisions:", vc_precision_soft, vc_precision_hard)
print("VC_soft, VC_hard Recalls:", vc_recall_soft, vc_recall_hard)
print("VC_soft, VC_hard F1-scores:", vc_f1_soft, vc_f1_hard)

######################
# Define the Generic Confusion Matrix Performance
######################

class_types = list(heart_df['ClassLabel'].unique())

from sklearn.metrics import confusion_matrix, classification_report

def conf_matrix(predictions, plot_title:str, save_path:str):
    ''' Plots conf. matrix and classification report '''
    cm=confusion_matrix(y_test, predictions)
    print("Classification Report:\n")
    cr=classification_report(y_test, predictions,
                             target_names=[class_types[i] for i in range(len(class_types))])
    print(cr)
    plt.figure(figsize=(6, 6))
    plt.title('Confusion Matrix: %s'%(plot_title))
    sns_hmp = sns.heatmap(cm, annot=True, xticklabels = [class_types[i] for i in range(len(class_types))],
                yticklabels = [class_types[i] for i in range(len(class_types))], fmt="d")
    fig = sns_hmp.get_figure()
    fig.savefig(save_path, dpi=150)
    
conf_matrix(voting_predictions_hard, 'Voting Classifier (Hard)', save_path='')
conf_matrix(adaboost_predictions, 'AdaBoost Classifier', save_path='')

###############################
# Decision Boundaries for Pairs of Feautres (Numerical)
###############################    

### Check the Effect of Voting Classifier

### select only the numerical cols

X_test_numeric = X_test[numerical_cols]

# Generating all pairs of numbers from 0 to 5
comb = combinations(np.arange(0, 5), 2)

# Using sets to obtain all unique combinations from 0 to 5 pairs
unique_combinations = set(comb)
labels = ['Healthy', 'Ill']


color_palette = plt.cm.cividis
plot_colors = ['g', 'r']
markers = ['*', 'o']
n_classes = len(y_train.unique())
plt.figure(figsize=(15, 10))

for pair_idx, pair in enumerate(sorted(unique_combinations)):
    # Only two corresponding features are taken each time
    X_test_cols = X_test_numeric.iloc[:, [pair[0], pair[1]]]

    # Creating and fitting the classifier to train data
    classifier = voting_classifier_hard.fit(X_test_cols, y_test)

    # Defining a grid of 5 columns and 2 rows
    ax = plt.subplot(2, 5, pair_idx + 1)
    # Plotting the pairs decision boundaries
    DecisionBoundaryDisplay.from_estimator(classifier,
                                           X_test_cols,
                                           cmap=color_palette,
                                           response_method="predict",
                                           ax=ax,
                                           xlabel=X_test_numeric.columns.to_list()[pair[0]],
                                           ylabel=X_test_numeric.columns.to_list()[pair[1]],
                                           alpha = 0.5)

    # Plotting the training points according to y_train class colors
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y_test == i)
        plt.scatter(X_test_cols.iloc[idx[:][0], 0],
                    X_test_cols.iloc[idx[:][0], 1],
                    c=color,
                    label=labels[i], marker=markers[i],
                    s=20, alpha=0.3)

plt.suptitle("Decision surface of Voting Classifier (Hard): Pairs of Numerical Features", fontsize=12)
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
# plt.show


###############
#Check GridSearch CV within Voting Classifier
##############

svc_classifier = SVC(probability=True) #kernel = rbf 

voting_classifier_hard = VotingClassifier(estimators=[('svm', svc_classifier), ('logreg', logreg_classifier), 
                                                      ('adaboost', adaboost_classifier)], voting='hard')


# 0.1, 0.5,
params = {'svm__C':[0.1, 0.5, 1, 30, 75,  100], 
          'svm__gamma':[0.005, 0.01, 0.05, 1, 10, 100], 
          'logreg__C':[0.05, 0.1, 0.5, 1, 30, 75, 100], 'adaboost__n_estimators':[20, 50, 70]}


for cv in tqdm(range(3, 6)):
    create_grid = GridSearchCV(estimator=voting_classifier_hard, param_grid=params, cv=cv)
    create_grid.fit(X_train, y_train)
    print ('score for %d fold CV := %3.2f'%(cv, create_grid.score(X_test, y_test)))
    print ('!!!!!!!! Best Params from Grid Search CV !!!!!!!!')
    print (create_grid.best_params_)

print ('Out of the Loop')


print ('grid CV best params: ', create_grid.best_params_) 
grid_CV_predictions = create_grid.predict(X_test)

conf_matrix(grid_CV_predictions, 'VC (Hard) with GridSearch (CV=5)', save_path='./GridCV_VC.png')

################
# libraries Used
################
'''
Matplotlib:  3.7.1
Numpy:  1.24.3
Scipy:  1.10.1
Pandas:  1.5.3
Seaborn:  0.12.2
sklearn:  1.3.0
'''
