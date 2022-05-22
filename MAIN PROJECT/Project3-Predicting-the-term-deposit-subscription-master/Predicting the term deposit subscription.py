# -*- coding: utf-8 -*-
"""
Created on Tue aug 23 01:45:40 2020

@author: nisar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

df=pd.read_csv("D:\Marketing Lead Classifier\Main project\Project3-Predicting-the-term-deposit-subscription-master\bank-full.csv")

df=df.iloc[:,[16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
#############################Data Exploration#################################

df.head() # To show first 5 rows of the data set

df.columns# To show the number of columns in the data set

df.tail()# To show last 5 rows of the data set

df.isnull().sum() #To check is their any null values in the features of the data set

df.shape # #To check number of observations(rows and columns)

df.dtypes # To check the data types of the features in data set

df.drop_duplicates(inplace=True)#Removing the duplicate values


df.info()#Give information about any null values/missing value

df.describe()#Give statistical information about the data set such as mean ,median ,mode,max value,min value

df.Target.value_counts()#To chech the count of target values

cormat=df.corr()#Helps to understand correlation between the dependent and independent variables

############################Data vizualization###########################################################

df.hist(figsize=(12,12)) #it shows the histogram plot for all features which are numeric

df.plot(kind='box',subplots=True,layout=(4,4),sharex=False,sharey=False,figsize=(18,18))
#it shows the boxplot for all features which are numeric

sns.pairplot(df) #Helps to get the scatter plot for all variables and to get pairs plot

fig= plt.figure(figsize=(12,12))
sns.heatmap(cormat,annot=True,cmap="BuGn_r")
plt.show()


###Countlot for various features in the dataset
sns.countplot(x='Target',data=df,palette='hls')
sns.countplot(x='job',data=df,palette='hls')
sns.countplot(x='marital',data=df,palette='hls')
sns.countplot(x='education',data=df,palette='hls')
sns.countplot(x='default',data=df,palette='hls')
sns.countplot(x='housing',data=df,palette='hls')
sns.countplot(x='loan',data=df,palette='hls')
sns.countplot(x='contact',data=df,palette='hls')
sns.countplot(x='month',data=df,palette='hls')
sns.countplot(x='poutcome',data=df,palette='hls')

#Since job,marital,education,default,housing,loan,contact,month,poutcome are categorical 
#we have to convrt them into numeric.So we use dummy_method to create the values

df=pd.get_dummies(df,columns=['job','marital','education','default','housing','loan','contact','month','poutcome'],drop_first=True)

#Since all the features in the data set are having different features .We have to apply scale function to normalize the data.

from sklearn.preprocessing import scale

################### Model Building step ####################################################################

########################LogisticRegression#######################################################print(accuracy_score(y_train,model.predict(X_train)))

from sklearn.model_selection import train_test_split

X=df.iloc[:,1:44]
y=df.iloc[:,0]
x=scale(X)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression 

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

n_errors_Log=print((y_pred!=y_test).sum())
cohen_kappa_score(y_test,y_pred)
print(accuracy_score(y_train,model.predict(X_train)))


########################Decision tree Classifier#######################################################print(accuracy_score(y_train,model.predict(X_train)))

from sklearn.tree import DecisionTreeClassifier

model1=DecisionTreeClassifier(criterion='entropy')
model1.fit(X_train,y_train)

y_pred1=model1.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))

n_errors_Dec=print((y_pred1!=y_test).sum())
cohen_kappa_score(y_test,y_pred1)
print(accuracy_score(y_train,model1.predict(X_train)))

#############################Random Forest Classifier#######################################################

from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier(n_estimators=100)
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))

n_errors_Ran=print((y_pred2!=y_test).sum())
cohen_kappa_score(y_test,y_pred2)
print(accuracy_score(y_train,model2.predict(X_train)))

###################################Extratreeclassifier####################################################

from sklearn.ensemble import ExtraTreesClassifier 

model3=ExtraTreesClassifier()
model3.fit(X_train,y_train)
y_pred3=model3.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(classification_report(y_test,y_pred3))
n_errors_ext=print((y_pred3!=y_test).sum())

cohen_kappa_score(y_test,y_pred3) 
print(accuracy_score(y_train,model3.predict(X_train)))

####################################Support Vector Machine ####################################################################################

from sklearn.svm import SVC  

model4=SVC()
model4.fit(X_train,y_train)

y_pred4=model4.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred4))
print(confusion_matrix(y_test,y_pred4))
print(classification_report(y_test,y_pred4))

n_errors_svc=print((y_pred4!=y_test).sum())
cohen_kappa_score(y_test,y_pred4)
print(accuracy_score(y_train,model4.predict(X_train)))

#################################Neural_Networks###############################################################################################################

from sklearn.neural_network import MLPClassifier 

model5=MLPClassifier(hidden_layer_sizes=(5,5))
model5.fit(X_train,y_train)
y_pred5=model5.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred5))
print(confusion_matrix(y_test,y_pred5))
print(classification_report(y_test,y_pred5))

n_errors_nn=print((y_pred5!=y_test).sum())
cohen_kappa_score(y_test,y_pred5)
print(accuracy_score(y_train,model5.predict(X_train)))

##############################Bagging Classifier##################################################################################################################


from sklearn.ensemble import BaggingClassifier 

model6=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'))
model6.fit(X_train,y_train)

y_pred6=model6.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred6))
print(confusion_matrix(y_test,y_pred6))
print(classification_report(y_test,y_pred6))


n_errors_BC=print((y_pred6!=y_test).sum())
cohen_kappa_score(y_test,y_pred6) 
print(accuracy_score(y_train,model6.predict(X_train)))



###############################Extreme Grdient Boosting Algorithm ########################################

import xgboost as xgb 

model7=xgb.XGBClassifier()
model7.fit(X_train,y_train)

y_pred7=model7.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred7))
print(confusion_matrix(y_test,y_pred7))
print(classification_report(y_test,y_pred7))


n_errors_BC=print((y_pred7!=y_test).sum())
cohen_kappa_score(y_test,y_pred7) 
print(accuracy_score(y_train,model7.predict(X_train)))

#################################CAT boost algorithm########################################################

from catboost import CatBoostClassifier

model8=CatBoostClassifier()
model8.fit(X_train,y_train)

y_pred8=model8.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred8))
print(confusion_matrix(y_test,y_pred8))
print(classification_report(y_test,y_pred8))


n_errors_CB=print((y_pred8!=y_test).sum())
cohen_kappa_score(y_test,y_pred8) 
print(accuracy_score(y_train,model8.predict(X_train)))



#########################################################################################################
#From all the above model results we can conclude that there is class imbalance probelm.
#Since we can observe training accuracy is more that testing accuracy , precision value
#is more for no class and recall value is less.We can observe majority class for no class and monority class for yes class
#So we need to solve class imbalance problem with the help of SMOTE

from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=444)
X_train_res,y_train_res=sm.fit_resample(X_train,y_train)

X_train_res.shape
y_train_res.shape
X_test.shape
y_test.shape

########################LogisticRegression after SMOTE#######################################################print(accuracy_score(y_train,model.predict(X_train)))


from sklearn.linear_model import LogisticRegression 

model9=LogisticRegression()
model9.fit(X_train_res,y_train_res)
y_pred9=model9.predict(X_test)
 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,y_pred9))#0.85
print(confusion_matrix(y_test,y_pred9))
print(classification_report(y_test,y_pred9))
n_errors_Log_SM=print((y_pred9!=y_test).sum())
cohen_kappa_score(y_test,y_pred9)
print(accuracy_score(y_train,model9.predict(X_train)))


########################Decision tree Classifier#######################################################print(accuracy_score(y_train,model.predict(X_train)))

from sklearn.tree import DecisionTreeClassifier

model10=DecisionTreeClassifier(criterion='entropy')
model10.fit(X_train_res,y_train_res)

y_pred10=model10.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred10))
print(confusion_matrix(y_test,y_pred10))
print(classification_report(y_test,y_pred10))

n_errors_Dec_SM=print((y_pred10!=y_test).sum())
cohen_kappa_score(y_test,y_pred10)
print(accuracy_score(y_train,model10.predict(X_train)))

#############################Random Forest Classifier#######################################################

from sklearn.ensemble import RandomForestClassifier
model11=RandomForestClassifier(n_estimators=100)
model11.fit(X_train_res,y_train_res)
y_pred11=model11.predict(X_test)


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test,y_pred11))
print(confusion_matrix(y_test,y_pred11))
print(classification_report(y_test,y_pred11))

n_errors_Ran_SM=print((y_pred11!=y_test).sum())
cohen_kappa_score(y_test,y_pred11)
print(accuracy_score(y_train,model11.predict(X_train)))

###################################Extratreeclassifier####################################################

from sklearn.ensemble import ExtraTreesClassifier 

model12=ExtraTreesClassifier()
model12.fit(X_train_res,y_train_res)
y_pred12=model2.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,y_pred12))
print(confusion_matrix(y_test,y_pred12))
print(classification_report(y_test,y_pred12))
n_errors_ext_SM=print((y_pred12!=y_test).sum())

cohen_kappa_score(y_test,y_pred12) 
print(accuracy_score(y_train,model12.predict(X_train)))

####################################Support Vector Machine ####################################################################################

from sklearn.svm import SVC  

model13=SVC()
model13.fit(X_train_res,y_train_res)

y_pred13=model13.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred13))
print(confusion_matrix(y_test,y_pred13))
print(classification_report(y_test,y_pred13))

n_errors_svc_SM=print((y_pred13!=y_test).sum())
cohen_kappa_score(y_test,y_pred13)
print(accuracy_score(y_train,model13.predict(X_train)))

#################################Neural_Networks###############################################################################################################

from sklearn.neural_network import MLPClassifier 

model14=MLPClassifier(hidden_layer_sizes=(5,5))
model14.fit(X_train_res,y_train_res)
y_pred14=model14.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred14))
print(confusion_matrix(y_test,y_pred14))
print(classification_report(y_test,y_pred14))

n_errors_nn_SM=print((y_pred14!=y_test).sum())
cohen_kappa_score(y_test,y_pred14)
print(accuracy_score(y_train,model14.predict(X_train)))

##############################Bagging Classifier##################################################################################################################


from sklearn.ensemble import BaggingClassifier 

model15=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'))
model15.fit(X_train_res,y_train_res)

y_pred15=model15.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred15))#0.74
print(confusion_matrix(y_test,y_pred15))
print(classification_report(y_test,y_pred15))


n_errors_BC_SM=print((y_pred15!=y_test).sum())
cohen_kappa_score(y_test,y_pred15) 
print(accuracy_score(y_train,model15.predict(X_train)))



###############################Extreme Grdient Boosting Algorithm ########################################

import xgboost as xgb 

model16=xgb.XGBClassifier()
model16.fit(X_train_res,y_train_res)

y_pred16=model16.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred16))#0.74
print(confusion_matrix(y_test,y_pred16))
print(classification_report(y_test,y_pred16))


n_errors_BC_SM=print((y_pred16!=y_test).sum())
cohen_kappa_score(y_test,y_pred16) #0.5220
print(accuracy_score(y_train,model16.predict(X_train)))

#################################CAT boost algorithm########################################################

from catboost import CatBoostClassifier

model17=CatBoostClassifier()
model17.fit(X_train_res,y_train_res)

y_pred17=model17.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred17))#0.74
print(confusion_matrix(y_test,y_pred17))
print(classification_report(y_test,y_pred17))


n_errors_CB_SM=print((y_pred17!=y_test).sum())
cohen_kappa_score(y_test,y_pred17) #0.52
print(accuracy_score(y_train,model17.predict(X_train)))


#################################################################################################################


####################################Linear SVC##################################################
from sklearn.svm import LinearSVC

model18= LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,C=0.55, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=42, max_iter=1000)

model18.fit(X_train,y_train)

y_pred18=model18.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred18))
print(confusion_matrix(y_test,y_pred18))
print(classification_report(y_test,y_pred18))


n_errors_LSVC=print((y_pred18!=y_test).sum())
cohen_kappa_score(y_test,y_pred18) 
print(accuracy_score(y_train,model18.predict(X_train)))

##################################Linear SVC _after smote##########################################

from sklearn.svm import LinearSVC

model19= LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,C=0.55, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=42, max_iter=1000)

model19.fit(X_train_res,y_train_res)

y_pred19=model19.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred19))
print(confusion_matrix(y_test,y_pred19))
print(classification_report(y_test,y_pred19))


n_errors_LSVCSM=print((y_pred19!=y_test).sum())
cohen_kappa_score(y_test,y_pred19) 
print(accuracy_score(y_train,model19.predict(X_train)))
#####################################################################################################


#From all the model results we can observe less errors for catboost classifier ,also cohen_kappa_score is 0.54 which is moderate compared to any other models
#So the finalised models is CatBoostClassifier


















































































