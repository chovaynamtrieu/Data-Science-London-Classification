import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import os

train_data = pd.read_csv('.\\train.csv',header = None)
train_labels = pd.read_csv('.\\trainLabels.csv',header = None)
test_data =  pd.read_csv('.\\test.csv',header = None)

from sklearn.preprocessing import StandardScaler, Normalizer
norm = Normalizer()
norm_train_data = norm.fit_transform(train_data)

# NAIBE BAYES
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
print('Naive Bayes',cross_val_score(nb_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 5)
print('KNN',cross_val_score(knn_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators = 100,random_state = 99)
print('Random Forest',cross_val_score(rfc_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver = 'saga')
print('Logistic Regression',cross_val_score(lr_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())

#SVM
from sklearn.svm import SVC
svc_model = SVC(gamma = 'auto')
print('SVM',cross_val_score(svc_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())

#DECISON TREE
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier()
print('Decision Tree',cross_val_score(dtree_model,norm_train_data, train_labels.values.ravel(), cv=10).mean())

#XGBOOST
from xgboost import XGBClassifier
xgb = XGBClassifier()
print('XGBoost',cross_val_score(xgb,norm_train_data, train_labels.values.ravel(), cv=10).mean())
