# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:03:02 2020

@author: sevi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data = pd.read_csv('mushrooms.csv')

data.head()

for cols in data.columns:
    if cols != 'class':
        temp = pd.get_dummies(data[cols],drop_first=True)
        data.drop(cols,axis=1,inplace=True)
        data = pd.concat([data,temp],axis=1)
data.head()

from sklearn.model_selection import train_test_split
X = data.drop('class',axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,rfc_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,rfc_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,rfc_pred))