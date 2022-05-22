# -*- coding: utf-8 -*-
"""
Created on Wed aug 24 22:07:33 2021

@author: nisar
"""


import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import scale

df=pd.read_csv("D:\\Marketing Lead Classifier\\Main project\\Project3-Predicting-the-term-deposit-subscription-master\\bank-full.csv")

df=df.iloc[:,[16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]

df=pd.get_dummies(df,columns=['job','marital','education','default','housing','loan','contact','month','poutcome'],drop_first=True)

from sklearn.model_selection import train_test_split

X=df.iloc[:,1:44]
y=df.iloc[:,0]
x=scale(X)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=444)
X_train_res,y_train_res=sm.fit_resample(X_train,y_train)

X_train_res.shape
y_train_res.shape
X_test.shape
y_test.shape

from catboost import CatBoostClassifier

model=CatBoostClassifier()
model.fit(X_train_res,y_train_res)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


n_errors_CB_SM=print((y_pred!=y_test).sum())
cohen_kappa_score(y_test,y_pred) 
print(accuracy_score(y_train,model.predict(X_train)))


pickle.dump(model,open('Model.pkl','wb'))




















