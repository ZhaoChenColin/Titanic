j#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:07:41 2017

@author: zhaochen
"""

import pandas as pd
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
x = pd.read_csv('~/Desktop/stonybrook/Fall2017/AMS595/finalProject-Team ZhaoChen & SihuiZong/dataset/train.csv')
y=x.pop("Survived")
print("original data")
print(x.describe())#there is missing date in Age, so we need to fill it up
x.head(10)
type(x['Age'][5])
#find which passenger class has the most missing value.
index=x['Age'].index[x.Age.apply(lambda x:numpy.isnan(x))]
xnan=x.ix[index]
xnan['Pclass'].value_counts()
#input age with mean
x["Age"].fillna(x.Age.mean(),inplace=True)
print("\n"+"data after I impute age with mean")
print(x.describe())
numeric_variables=list(x.dtypes[x.dtypes != "object"].index)
print(x[numeric_variables].head())
model=RandomForestRegressor(n_estimators=100,oob_score=True,random_state=42)
model.fit(x[numeric_variables],y)
print ("C-stat: ", roc_auc_score(y, model.oob_prediction_))

#categorical variables
x.drop(["Name","Ticket","PassengerId"],axis=1,inplace=True)
print(x.head())
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"
x["Cabin"]=x.Cabin.apply(clean_cabin)
categorical_variables=['Sex','Cabin','Embarked']
for variable in categorical_variables:
    x[variable].fillna('Missing',inplace=True)
    dummies=pd.get_dummies(x[variable],prefix=variable)
    x=pd.concat([x,dummies],axis=1)
    x.drop([variable],axis=1,inplace=True)

model=RandomForestRegressor(100,oob_score=True,n_jobs=-1,random_state=42)
model.fit(x,y)
print ("C-stat: ", roc_auc_score(y, model.oob_prediction_))
print(model.feature_importances_)
#feature_importances=pd.Series(model.feature_importances_,index=x.columns)
#feature_importances.plot(kind="barh",figsize=(7,6))

#the number of trees in the forest.Choose as high of a number as your computer can handle
results=[]
n_estimator_options=[30,50,100,200,500,1000,2000]
for trees in n_estimator_options:
    model=RandomForestRegressor(trees,oob_score=True,n_jobs=-1,random_state=42)
    model.fit(x,y)
    print(trees,"trees")
    roc=roc_auc_score(y,model.oob_prediction_)
    print("C-stat: ",roc)
    results.append(roc)
    print("")
#pd.Series(results,n_estimator_options).plot()
#choose tree=1000
#max_features:the number of features to consider when looking for the best split
results=[]
max_features_options=["auto",None,"sqrt","log2",0.9,0.2]
for max_features in max_features_options:
    model=RandomForestRegressor(n_estimators=1000,oob_score=True,n_jobs=-1,random_state=42,max_features=max_features)
    model.fit(x,y)
    print(max_features,"option")
    roc=roc_auc_score(y,model.oob_prediction_)
    print("C-stat: ",roc)
    results.append(roc)
    print("")
#pd.Series(results,max_features_options).plot(kind="barh",xlim=(0.85,0.88))
#min_sample_leaf:the minimum number of samples in newly created leaves.
results=[]
min_samples_leaf_options=[1,2,3,4,5,6,7,8,9,10]
for min_samples in min_samples_leaf_options:
    model=RandomForestRegressor(n_estimators=1000,oob_score=True,n_jobs=-1,random_state=42,max_features="auto",min_samples_leaf=min_samples)
    model.fit(x,y)
    print(min_samples,"min samples")
    roc=roc_auc_score(y,model.oob_prediction_)
    print("C-stat: ",roc)
    results.append(roc)
    print("")
pd.Series(results,min_samples_leaf_options).plot()
#we choose min_sample_leaf = 5
#final model
model=RandomForestRegressor(n_estimators=1000,oob_score=True,n_jobs=-1,random_state=42,max_features="auto",min_samples_leaf=5)
model.fit(x,y)
roc=roc_auc_score(y,model.oob_prediction_)
print("C-stat: ",roc)
