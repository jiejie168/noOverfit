__author__ = 'Jie'
"""
This code is used to fit a suitable model for a training set with only 250 samples, but with 300 variables.
there is no doubt that overfitting will occur via using normal ML method as before.
# the fit accuracy is not good, but the method can be used for other cases.
"""
##  experiment 1, reference from Chris.
############################################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
# randomly create a dataframe (250*300), and a target, check the auc
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RepeatedKFold,RepeatedStratifiedKFold,KFold,GridSearchCV,StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score,make_scorer
##############################################################################################

##############################################################################################
# read training data and test data
trains= pd.read_csv("D:/python-ml/kaggles_competition/noOverfit/train.csv")
tests= pd.read_csv("D:/python-ml/kaggles_competition/noOverfit/test.csv")
##############################################################################################

##############################################################################################
#### some constant parameters
random_seed = 234587
cv=20
repeats=1
splits=20
steps=15
test_size=0.15
##############################################################################################


# there should be a check and visulization of the data.
#######################################

##############################################################################################
# since the data are clean, there is no need to further clean and manipulate the data
## first obtain the training data
columns=trains.columns
index=trains.index
X_train=trains.drop(['id','target'],axis=1)  # axis=1: drop columns
X_test=tests.drop(['id'],axis=1)  # axis=1: drop columns
y_train=trains['target']
##############################################################################################

##############################################################################################
# First, we make a simple LogisticRegression accounting for cross-validation
# for LR, it is necessary to have a normalization
# use statistics that are robust to outliers.
scaler=RobustScaler()
X_train=pd.DataFrame(scaler.fit_transform(X_train))  #  scaler return  the numpy.ndarray, then change to df
X_test=pd.DataFrame(scaler.fit_transform(X_test))

# print (type(X_train))
##############################################################################################

##############################################################################################
# add some statistic features for the dataset
from sklearn.neighbors import NearestNeighbors
def add_data(X):
    # X should be a dataframe
    # return a dataframe
    added_df=pd.DataFrame()
    added_df['mean']=X.mean(axis=1)
    added_df['median']=X.median(axis=1) # median value of every row, value corresponding to CDF=0.5
    added_df['kurt']=X.kurt(axis=1) #  kurtosis values, E[(X-uX)4]
    added_df['skew']=X.skew(axis=1) # skewness values, E[(X-uX)3]
    added_df['max']=X.max(axis=1)
    added_df['min']=X.min(axis=1)
    added_df['mad']=X.mad(axis=1) # mean absolute value of every row

    # find some nearest points (5 points here) for the given dataset.
    # use these distances, to sythesis some other data
    neighbors=NearestNeighbors(n_neighbors=5)
    neighbors.fit(X)
    distances, indices=neighbors.kneighbors(X)
    distances=np.delete(distances,0,axis=1) # delete the first column, point itself
    added_df['minDist']=distances.min(axis=1)
    added_df['maxDist']=distances.max(axis=1)
    added_df['meanDist']=distances.mean(axis=1)

    df_all=pd.concat([pd.DataFrame(X),added_df],axis=1)
    return df_all
##############################################################################################

##############################################################################################
#### start the simple grid search for the model
#### define a scoring method
## define a self_scoring method for grid search
def score_method(y_true,y_score):
    try:
        return roc_auc_score(y_true,y_score)
    except:
        return 0.5
myScorer=make_scorer(score_method)

params={'C':[.2, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37],
        'tol':[0.0001, 0.00011, 0.00009]}
# Exhaustive search over specified parameter values for an estimator.
# this estimator will be used for the following feature selector.
X_train=add_data(X_train).values  # change to numpy array, for the following use
X_test=add_data(X_test).values
# print(X_train.head())
clf=LogisticRegression(random_state=42,penalty='l1',solver='liblinear',C=0.31,class_weight='balanced',)
gridsearch=GridSearchCV(clf,param_grid=params,scoring=myScorer,cv=20, verbose=0,)
gridsearch.fit(X_train,y_train)
##############################################################################################

#####################################################################################################
## in order to have a better result, we further search using the feature importance selection strategy
# meanwhile, we split the traindata as well to loop so that largely reduced overfitting.
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error,mean_absolute_error,roc_auc_score,r2_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# divide the training dataset into splits folds, using len(splits)-1 for training, and 1 for validation.
# repeat such dividing for 'repeats' times. The overall outputs fit model is splits*repeats
rskf=RepeatedStratifiedKFold(n_splits=splits,n_repeats=repeats,random_state=random_seed)
# rskf=StratifiedShuffleSplit(n_splits=splits, test_size=test_size,
#                        random_state=random_seed)
# select the important features for a specific estimator.create a feature selector
feature_selector=RFECV(gridsearch.best_estimator_,step=steps,cv=cv,scoring=myScorer)

# start loop for the model fitting for every single split. total loop number: splits*repeats
counter=0
predictions=pd.DataFrame()
print("counter,mse, mae, auc,r2, grid_search.best_score_, feature_selector.n_features_, message ")
for train_index, val_index in rskf.split(X_train,y_train):
    # train a model for every single split.
    #select train_index rows of data for training
    X,X_val=X_train[train_index], X_train[val_index]
    y,y_val=y_train[train_index], y_train[val_index]

    # # fit the RFE model and automatically tune the number of selected.
    feature_selector.fit(X,y)
    X_import=feature_selector.transform(X) # reduce X to the selected features
    X_val_import=feature_selector.transform(X_val)
    X_test_import=feature_selector.transform(X_test)

    # fit the model use the reduced selected features.
    # calculate the prediction use the fitted model with the selected features.
    gridsearch=GridSearchCV(feature_selector.estimator_,param_grid=params,scoring=myScorer,cv=cv)
    gridsearch.fit(X_import,y)
    # calculate the probability of Validation data, and select the latter value which indicates the probability to be 1
    y_pred_val=gridsearch.best_estimator_.predict_proba(X_val_import)[:,1]

    mse=mean_squared_error(y_val,y_pred_val)
    mae=mean_absolute_error(y_val,y_pred_val)
    r2=r2_score(y_val,y_pred_val)
    auc=roc_auc_score(y_val,y_pred_val)
    # val_cos = cosine_similarity(y_val.values.reshape(1, -1), y_pred_val.reshape(1, -1))[0][0]
    # val_dst = euclidean_distances(y_val.values.reshape(1, -1), y_pred_val.reshape(1, -1))[0][0]
    # add the fit model into the final predictions, here only based on the r2_score
    threshold =0.185
    if r2>=threshold:
        message= "<---OK"
        prediction= gridsearch.best_estimator_.predict_proba(X_test_import)[:,1]
        predictions=pd.concat([predictions,pd.DataFrame(prediction)],axis=1) # use column as the axis, joint all the dataframe
    else:
        message="<---Skip"
    counter+=1
    print("{:2} | {:.4f}|  {:.4f}   |  {:.4f}|  {:.4f}  |  {:.4f}|  {:3} {}  ".format(counter,
    mse, mae, auc, r2, gridsearch.best_score_, feature_selector.n_features_, message))
##############################################################################################


##############################################################################################
# predictions results output and  save !
# the final predictions should be the average values of all the fitted models
print ("there are {} out of {} models are selected for average".format(len(predictions.columns),splits*repeats))
# mean_predictions=predictions.mean(axis=1) #  calculate the mean value of each row
mean_predictions=pd.DataFrame(predictions.mean(axis=1) ) #  should do another creation for a dataFrame. otherwise, it is only a series.
mean_predictions.index+=250  # the test data is started from 250
mean_predictions.columns=['target']
mean_predictions.to_csv("D:/python-ml/kaggles_competition/noOverfit/myResult.csv",index_label='id',index=True)
print ('completed !')
##############################################################################################