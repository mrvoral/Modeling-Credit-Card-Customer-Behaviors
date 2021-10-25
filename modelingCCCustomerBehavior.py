# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 13:55:31 2020

@author: merve
"""

import numpy as np
import pandas as pd
import time
tic = time.time()
#import the data and the labels
df = pd.read_csv("hw08_training_data.csv")
df.fillna(df.median(), inplace=True)

columns=list(df.select_dtypes(exclude=["number","bool_"]))
for i in columns: 
    dfDummies = pd.get_dummies(df[i], prefix = 'category')
    df = pd.concat([df, dfDummies], axis=1)
    df = df.drop(columns=i)
    
X_test=pd.read_csv("hw08_test_data.csv")
X_test.fillna(X_test.median(), inplace=True)

#encoding test set
columns=list(X_test.select_dtypes(exclude=["number","bool_"]))
for i in columns: 
    X_testDummies = pd.get_dummies(X_test[i], prefix = 'category')
    X_test = pd.concat([X_test, X_testDummies], axis=1)
    X_test = X_test.drop(columns=i)



#import labels
Labels = pd.read_csv("hw08_training_label.csv")

Y=np.array(Labels)[:,1]
locs=np.isnan(Y) #locations of nans
X=df.drop(df.index[locs], axis=0)
Y=Y[np.invert(locs)]

Y2=np.array(Labels)[:,2]
locs2=np.isnan(Y2) #locations of nans
X2=df.drop(df.index[locs2], axis=0)
Y2=Y2[np.invert(locs2)]

Y3=np.array(Labels)[:,3]
locs3=np.isnan(Y3) #locations of nans
X3=df.drop(df.index[locs3], axis=0)
Y3=Y3[np.invert(locs3)]

Y4=np.array(Labels)[:,4]
locs4=np.isnan(Y4) #locations of nans
X4=df.drop(df.index[locs4], axis=0)
Y4=Y4[np.invert(locs4)]

Y5=np.array(Labels)[:,5]
locs5=np.isnan(Y5) #locations of nans
X5=df.drop(df.index[locs5], axis=0)
Y5=Y5[np.invert(locs5)]

Y6=np.array(Labels)[:,6]
locs6=np.isnan(Y6) #locations of nans
X6=df.drop(df.index[locs6], axis=0)
Y6=Y6[np.invert(locs6)]


# train a random forest
#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=120,max_depth=10,class_weight="balanced")
rfc2=RandomForestClassifier(n_estimators=130,max_depth=10,class_weight="balanced")
rfc3=RandomForestClassifier(n_estimators=130,max_depth=9,class_weight="balanced")
rfc4=RandomForestClassifier(n_estimators=140,max_depth=9,class_weight="balanced")
rfc5=RandomForestClassifier(n_estimators=120,max_depth=9,class_weight="balanced")
rfc6=RandomForestClassifier(n_estimators=140,max_depth=10,class_weight="balanced")


#grid search
#param_grid = {'n_estimators': [80,100], 'max_depth': list(np.arange(5,8))}
#rfc = GridSearchCV(rfc, param_grid,scoring=('roc_auc'), cv=10)

#fit and evaluate posterior for training set
Y_rf = rfc.fit(X, Y)
Y_pred=Y_rf.predict_proba(X)[:,1]

Y_rf2 = rfc2.fit(X2, Y2)
Y_pred2=Y_rf2.predict_proba(X2)[:,1]

Y_rf3 = rfc2.fit(X3, Y3)
Y_pred3=Y_rf2.predict_proba(X3)[:,1]

Y_rf4 = rfc2.fit(X4, Y4)
Y_pred4=Y_rf4.predict_proba(X4)[:,1]

Y_rf5 = rfc5.fit(X5, Y5)
Y_pred5=Y_rf5.predict_proba(X5)[:,1]

Y_rf6 = rfc6.fit(X6, Y6)
Y_pred6=Y_rf6.predict_proba(X6)[:,1]


#rfc.best_params_
#calculate mean auroc using 10-fold cross validation
from sklearn.model_selection import cross_validate
scores = cross_validate(rfc, X, Y, cv=10, scoring=('roc_auc'), return_train_score=True)
print("Mean AUROC for target 1 is:")
print(np.mean(scores['test_score']))
scores2 = cross_validate(rfc2, X2, Y2, cv=10, scoring=('roc_auc'), return_train_score=True)
print("Mean AUROC for target 2 is:")
print(np.mean(scores2['test_score']))
scores3 = cross_validate(rfc3, X3, Y3, cv=10, scoring=('roc_auc'), return_train_score=True)
print("Mean AUROC for target 3 is:")
print(np.mean(scores3['test_score']))

scores4 = cross_validate(rfc4, X4, Y4, cv=10, scoring=('roc_auc'), return_train_score=True)
print("Mean AUROC for target 4 is:")
print(np.mean(scores4['test_score']))
scores5 = cross_validate(rfc5, X5, Y5, cv=10, scoring=('roc_auc'), return_train_score=True)
print("Mean AUROC for target 5 is:")
print(np.mean(scores5['test_score']))
scores6 = cross_validate(rfc6, X6, Y6, cv=10, scoring=('roc_auc'), return_train_score=True)
print("Mean AUROC for target 6 is:")
print(np.mean(scores6['test_score']))

del df, X,X2,X3,X4,X5,X6
## evaluate posteriors on the test set
Y_test=Y_rf.predict_proba(X_test)[:,1].reshape((-1,1))
Y_test2=Y_rf2.predict_proba(X_test)[:,1].reshape((-1,1))
Y_test3=Y_rf3.predict_proba(X_test)[:,1].reshape((-1,1))
Y_test4=Y_rf4.predict_proba(X_test)[:,1].reshape((-1,1))
Y_test5=Y_rf5.predict_proba(X_test)[:,1].reshape((-1,1))
Y_test6=Y_rf6.predict_proba(X_test)[:,1].reshape((-1,1))

##concatanate with ID
Y_o=np.concatenate((np.array(X_test["ID"]).reshape((-1,1)),Y_test,Y_test2,Y_test3,Y_test4,Y_test5,Y_test6),1)

## write predictions to csv files
pd.DataFrame(Y_o).to_csv("hw08_test_predictions.csv",header=["ID","TARGET_1","TARGET_2","TARGET_3","TARGET_4","TARGET_5","TARGET_6"],index=None)
#
## plot ROC curve
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(Y, Y_pred)
fpr2, tpr2, threshold = metrics.roc_curve(Y2, Y_pred2)
fpr3, tpr3, threshold = metrics.roc_curve(Y3, Y_pred3)
fpr4, tpr4, threshold = metrics.roc_curve(Y4, Y_pred4)
fpr5, tpr5, threshold = metrics.roc_curve(Y5, Y_pred5)
fpr6, tpr6, threshold = metrics.roc_curve(Y6, Y_pred6)


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic for target 1,2,3,4,5,6')
plt.plot(fpr, tpr, 'b')
plt.plot(fpr2, tpr2, 'r')
plt.plot(fpr3, tpr3, 'g')
plt.plot(fpr4, tpr4, 'c')
plt.plot(fpr5, tpr5, 'm')
plt.plot(fpr6, tpr6, 'y')
plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(('target 1','target 2','target 3','target 4','target 5','target 6','random classifer'))
plt.show()
