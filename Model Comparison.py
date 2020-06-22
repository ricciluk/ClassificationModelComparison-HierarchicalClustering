# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:22:47 2020

@author: ricci
"""

import pandas as pd
import numpy as np
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import seaborn as sns #visualization
from sklearn import metrics
import matplotlib.pyplot as plt 
import dexplot as dxp #percentage plot
from scipy.stats import chisquare, pearsonr
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn import preprocessing

organics_dummy=pd.read_csv('C:/Users/ricci/Desktop/Coding & Techniques/GitHub/Classification & Clustering/organics_dummy.csv')
organics_dt=pd.read_csv('C:/Users/ricci/Desktop/Coding & Techniques/GitHub/Classification & Clustering/organics_dt.csv')
#%%
#Up-sample using SMOTE
def data_smote(data):
    us = SMOTE(random_state=0)
    X = data.loc[:,data.columns!='TargetBuy']
    y = data['TargetBuy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #only up-sample the training data and use the original test data for testing
    us_data_X,us_data_y=us.fit_sample(X_train, y_train)
    us_data_X = pd.DataFrame(data=us_data_X,columns=X_train.columns)
    us_data_y= pd.DataFrame(data=us_data_y,columns=['TargetBuy'])
    #sns.countplot(x='TargetBuy',data=us_data_y,palette='hls')
    return us_data_X,us_data_y,X_test,y_test

#####Logistic Regression
def log_model(train_x,train_y,test_x,test_y):
    logreg = LogisticRegression()
    
    #if variable selection needed
    drop_list=['PromSpend','Female']
    train_x=train_x.drop(columns=drop_list,axis=1)
    test_x=test_x.drop(columns=drop_list,axis=1)
    
    logreg.fit(train_x, train_y.values.ravel())
#    print(sm.Logit(train_y,train_x).fit().summary2())
#    logreg.coef_   
    fpr_log, tpr_log = roc_curve(test_y, logreg.predict_proba(test_x)[:,1])[:2]
    logit_roc_auc=roc_auc_score(test_y, logreg.predict(test_x))
    return logreg.score(train_x,train_y),logreg.score(test_x,test_y),fpr_log,tpr_log,logit_roc_auc

log_model(data_smote(organics_dummy)[0],data_smote(organics_dummy)[1],data_smote(organics_dummy)[2],data_smote(organics_dummy)[3])

#####Tree-based Models
#Random Forest
def random_forest(train_x,train_y,test_x,test_y):
    max_features = list(range(3,6)) #3 is the best result
    n_estimators = list(range(90,110))
    
    rf_result=pd.DataFrame(columns=['Number of Variables', 'Number of Trees','Mis_Rate'])
    for n_estimator in n_estimators:
        for max_feature in max_features:
            rf = RandomForestClassifier(n_estimators=n_estimator,max_features=max_feature, max_depth=6,random_state=1)
            rf.fit(train_x, train_y.values.ravel())
            rf_mrate = 1-metrics.accuracy_score(rf.predict(test_x), test_y)
            rf_result=rf_result.append({'Number of Variables':max_feature,'Number of Trees':n_estimator,'Mis_Rate':rf_mrate},ignore_index=True)
    rf_result=rf_result.sort_values(['Mis_Rate'],ascending=True).reset_index(drop=True)
    
    rf = RandomForestClassifier(n_estimators=int(rf_result.iloc[0,1]),max_features=int(rf_result.iloc[0,0]), max_depth=6,random_state=1)
    rf.fit(train_x, train_y.values.ravel())
    fpr_rf, tpr_rf = roc_curve(test_y, rf.predict_proba(test_x)[:,1])[:2]
    rf_roc_auc=roc_auc_score(test_y, rf.predict(test_x))   

    return rf.score(train_x,train_y),rf.score(test_x,test_y),fpr_rf,tpr_rf,rf_roc_auc       

random_forest(data_smote(organics_dt)[0],data_smote(organics_dt)[1],data_smote(organics_dt)[2],data_smote(organics_dt)[3])[:2]


#####Tree-based Models
#xgboost
import xgboost as xgb
def xgboost(train_x,train_y,test_x,test_y):
    xgb_default_params = {
    'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 10, 'subsample': 0.8,
    'colsample_bynode': 0.8, 'colsample_bylevel' : 1,'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'base_score': 0.5, 'gamma': 0, 'max_delta_step': 0, 'scale_pos_weight': 1,
    'objective': 'binary:logistic', 'booster': None
    }
    
    xgb_model = xgb.XGBClassifier(**xgb_default_params)
    
    bst = xgb_model.fit(train_x,train_y.values.ravel(),early_stopping_rounds=200, eval_set=[(train_x,train_y),(test_x,test_y)], eval_metric = 'auc')
    fpr_xgbst, tpr_xgbst = roc_curve(test_y, bst.predict_proba(test_x)[:,1])[:2]
    auc_xgbt=roc_auc_score(test_y, bst.predict(test_x))
    
    return bst.score(train_x,train_y),bst.score(test_x,test_y),fpr_xgbst,tpr_xgbst,auc_xgbt

xgboost(data_smote(organics_dt)[0],data_smote(organics_dt)[1],data_smote(organics_dt)[2],data_smote(organics_dt)[3])

#lightgbt
import lightgbm as lgb
def lightgbt(train_x,train_y,test_x,test_y):
    lgb_default_params = {
        'boosting_type': 'gbdt', 'n_estimators': 105, 'learning_rate': 0.1, 'max_depth': 6, 'num_leaves': 60, 'min_child_samples': 30,
         'min_child_weight': 0.001, 'colsample_bytree': 0.8, 'min_split_gain': 0.0, 'objective': None, 'subsample': 0.8,
         'subsample_freq': 4, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'silent': True, 'random_state': None, 'class_weight': None
     }

    clf = lgb.LGBMClassifier(**lgb_default_params)
    bst = clf.fit(train_x,train_y.values.ravel(), eval_set=[(train_x,train_y),(test_x,test_y)], eval_metric = 'auc')
    fpr_lightbst, tpr_lightbst = roc_curve(test_y, bst.predict_proba(test_x)[:,1])[:2]
    
    return bst.score(train_x,train_y),bst.score(test_x,test_y),fpr_lightbst,tpr_lightbst,bst.best_score_['valid_1']['auc']

lightgbt(data_smote(organics_dt)[0],data_smote(organics_dt)[1],data_smote(organics_dt)[2],data_smote(organics_dt)[3])


#Neural Network

#normalize the data
def nml_data(train_x,test_x):
    nml_train=pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(train_x))
    nml_train.columns=train_x.columns
    
    nml_test=pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(test_x))
    nml_test.columns=test_x.columns

    return nml_train,nml_test

from keras.models import Sequential
from keras.layers import Dense
def nn_clf(train_x,train_y,test_x,test_y):
    nn_model=Sequential()
    nn_model.add(Dense(30,input_dim=32,activation='relu')) #input_dim: number of variables, 30 nodes for the first hidden layer
    #ReLU: rectified linear unit -> y=max(0,x)
    nn_model.add(Dense(15, activation='relu'))
#    nn_model.add(Dense(15, activation='relu'))
#    nn_model.add(Dense(15, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))
    #sigmoid: easy to map the expected result of probability between 0 and 1
    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #Cross Entropy: Loss Function for Classification Models
    #Optimizer: 'adam' stochastic gradient descent algorithm
    
    nn_model.fit(train_x, train_y, epochs=10, batch_size=5)
    #epoch: run a fixed number of iterations through the data set
    #batch_size(gradient descent): number of samples passed to be trained for updating the algorithm and improving performance;
    
    nn_roc_auc=roc_auc_score(test_y, nn_model.predict_classes(test_x))
    fpr_nn, tpr_nn = roc_curve(test_y,nn_model.predict_proba(test_x))[:2]
    
    return nn_model.evaluate(train_x, train_y),nn_model.evaluate(test_x,test_y),fpr_nn,tpr_nn,nn_roc_auc

nn_clf(nml_data(data_smote(organics_dummy)[0],data_smote(organics_dummy)[2])[0],data_smote(organics_dummy)[1],nml_data(data_smote(organics_dummy)[0],data_smote(organics_dummy)[2])[1],data_smote(organics_dummy)[3])

#combine result and plot ROC
def model_iteration(data_dummy,data_nodummy):
    roc_all=pd.DataFrame(columns=['model','fpr','tpr','auc'])
#    model_list1=['NeuralNetwork']
    model_list1=['LogisticRegression','NeuralNetwork']
    model_list2=['RandomForest','lgb','xgboost']
    for i in model_list1:
        train_x,train_y,test_x,test_y=data_smote(data_dummy)[0],data_smote(data_dummy)[1],data_smote(data_dummy)[2],data_smote(data_dummy)[3]
        if i == 'LogisticRegression' :
            roc=pd.DataFrame(columns=['model','fpr','tpr','auc'])
            roc['fpr'],roc['tpr'],roc['auc']=log_model(train_x,train_y,test_x,test_y)[2:5]
            roc['model']=[i]*len(roc)
            roc_all=pd.concat([roc_all,roc],axis=0)
        else :
            roc=pd.DataFrame(columns=['model','fpr','tpr','auc'])
            roc['fpr'],roc['tpr'],roc['auc']=nn_clf(nml_data(train_x,test_x)[0],train_y,nml_data(train_x,test_x)[1],test_y)[2:5]
            roc['model']=[i]*len(roc)
            roc_all=pd.concat([roc_all,roc],axis=0)
            break
    for i in model_list2:
        train_x,train_y,test_x,test_y=data_smote(data_nodummy)[0],data_smote(data_nodummy)[1],data_smote(data_nodummy)[2],data_smote(data_nodummy)[3]
        if i == 'RandomForest' :
            roc=pd.DataFrame(columns=['model','fpr','tpr','auc'])
            roc['fpr'],roc['tpr'],roc['auc']=random_forest(train_x,train_y,test_x,test_y)[2:5]
            roc['model']=[i]*len(roc)
            roc_all=pd.concat([roc_all,roc],axis=0)
        elif i == 'lgb' :
            roc=pd.DataFrame(columns=['model','fpr','tpr','auc'])
            roc['fpr'],roc['tpr'],roc['auc']=lightgbt(train_x,train_y,test_x,test_y)[2:5]
            roc['model']=[i]*len(roc)
            roc_all=pd.concat([roc_all,roc],axis=0)
        elif i == 'xgboost' :
            roc=pd.DataFrame(columns=['model','fpr','tpr','auc'])
            roc['fpr'],roc['tpr'],roc['auc']=xgboost(train_x,train_y,test_x,test_y)[2:5]
            roc['model']=[i]*len(roc)
            roc_all=pd.concat([roc_all,roc],axis=0)
            
    return roc_all

roc_all=model_iteration(organics_dummy,organics_dt)
    

#ROC
plt.figure(figsize=(20,20))
sns.lineplot(x='fpr',y='tpr',color="blue",data=roc_all[roc_all['model']=='LogisticRegression'],label='Logistic Regression (area = %0.2f)' % roc_all[roc_all['model']=='LogisticRegression']['auc'][0])
sns.lineplot(x='fpr',y='tpr',color="orange",data=roc_all[roc_all['model']=='NeuralNetwork'],label='NeuralNetwork (area = %0.2f)' % roc_all[roc_all['model']=='NeuralNetwork']['auc'][0])
sns.lineplot(x='fpr',y='tpr',color="green",data=roc_all[roc_all['model']=='RandomForest'],label='RandomForest (area = %0.2f)' % roc_all[roc_all['model']=='RandomForest']['auc'][0])
sns.lineplot(x='fpr',y='tpr',color="red",data=roc_all[roc_all['model']=='lgb'],label='lgb (area = %0.2f)' % roc_all[roc_all['model']=='lgb']['auc'][0])
sns.lineplot(x='fpr',y='tpr',color="purple",data=roc_all[roc_all['model']=='xgboost'],label='xgboost (area = %0.2f)' % roc_all[roc_all['model']=='xgboost']['auc'][0])
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")


