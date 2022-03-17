import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import os

train_data = pd.read_csv('.\\train.csv',header = None)
train_labels = pd.read_csv('.\\trainLabels.csv',header = None)
test_data =  pd.read_csv('.\\test.csv',header = None)

#Train-Test Split
x_train,x_test,y_train,y_test = train_test_split(train_data,train_labels, test_size = 0.30, random_state = 101)


# NAIBE BAYES
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train,y_train.values.ravel())
predicted= model.predict(x_test)
print('Naive Bayes',accuracy_score(y_test, predicted))

#KNN
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model.fit(x_train,y_train.values.ravel())
predicted= knn_model.predict(x_test)
print('KNN',accuracy_score(y_test, predicted))

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(n_estimators = 100,random_state = 99)
rfc_model.fit(x_train,y_train.values.ravel())
predicted = rfc_model.predict(x_test)
print('Random Forest',accuracy_score(y_test,predicted))

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver = 'saga')
lr_model.fit(x_train,y_train.values.ravel())
lr_predicted = lr_model.predict(x_test)
print('Logistic Regression',accuracy_score(y_test, lr_predicted))

#SVM
from sklearn.svm import SVC

svc_model = SVC(gamma = 'auto')
svc_model.fit(x_train,y_train.values.ravel())
svc_predicted = svc_model.predict(x_test)
print('SVM',accuracy_score(y_test, svc_predicted))

#DECISON TREE
from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier()
dtree_model.fit(x_train,y_train.values.ravel())
dtree_predicted = dtree_model.predict(x_test)
print('Decision Tree',accuracy_score(y_test, dtree_predicted))

#XGBOOST
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(x_train,y_train.values.ravel())
xgb_predicted = xgb.predict(x_test)
print('XGBoost',accuracy_score(y_test, xgb_predicted))
