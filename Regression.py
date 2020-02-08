#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:02:35 2019

@author: yiliu
"""
#linear regression, KNN, adaboost, random forest, SVM
#Do K-fold cross validation for both.
#For regression show: R 2 , Adjusted R 2 , RMSE, correlation matrix, p-values of
#independent variables(in Linear regression) (codes 10)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',30)
plt.interactive(False)

df=pd.read_excel(r"....\ENB2012_data.xlsx")
columns=['Relative_Compactness','Surface_Area','Wall_Area','Roof_Area','Overall_Height',
'Orientation','Glazing_Area','Glazing_Area_Distribution','Heating_Load','Cooling_Load']
df.columns=columns
df.info()
df.describe()

X=df.drop(['Heating_Load','Cooling_Load'],axis=1)
y1=df['Heating_Load']
y2=df['Cooling_Load']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y1,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

sns.distplot(y1)
plt.show()

#Compare different Regression models
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

#Correlation heatmap
df1=df.drop('Cooling_Load',axis=1)
cor=df1.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.Reds,fmt='.2f')
plt.xticks(rotation=20)
plt.show()

#Linear Regression 
X_trainnew = np.append (arr=np.ones([X_train.shape[0],1]).astype(int), values = X_train, axis = 1)
import statsmodels.formula.api as sm
leng=X_trainnew.shape[1]
X_opt=list(range(0,leng))
regressor = sm.OLS(y_train, X_trainnew[:,X_opt]).fit()
print(regressor.summary())

#get rid of 'Oritentation' column since the P value >0.05
X_train=np.delete(X_train,5,1)
X_test=np.delete(X_test,5,1)

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
r2_score=r2_score(y_test,y_pred)
adj_r2=1-(1-r2_score)*(y_test.shape[0]-1)/(y_test.shape[0]-X_train.shape[1]-1)
print('R2 score of Linear Regression:', r2_score)
print('Adjusted R2 score:',adj_r2)
print('MSE of Linear Regression: ', mean_squared_error(y_test,y_pred))
print('Cross validation score (cv=4) of Linear Regression:',cross_val_score(model,X,y1,cv=4).mean())


#KNN
score=[]
for k in range(1,20):
    model=KNeighborsRegressor(n_neighbors=k,p=2,metric='minkowski')
    model.fit(X,y1)
    score.append(cross_val_score(model,X,y1,cv=4).mean())
print('K with max r2_score is:',score.index(max(score))+1)

#score plot of k(1~19)
plt.bar(x=range(1,20),height=score)
plt.show()

model=KNeighborsRegressor(n_neighbors=1,p=2,metric='minkowski')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('MSE of KNN: ', mean_squared_error(y_test,y_pred))
print('Cross validation score (cv=4) of KNN:',cross_val_score(model,X,y1,cv=4).mean())

#Adaboost
model=AdaBoostRegressor()
para_dict={
           'n_estimators':[300,350,400,450],
           'learning_rate':[0.5,1,2,4,6]
           }
clf=GridSearchCV(model,para_dict,cv=4,scoring='r2')
clf.fit(X,y1)
clf.best_params_


model=AdaBoostRegressor(n_estimators=350 , learning_rate=2 )
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('MSE of Adaboost: ', mean_squared_error(y_test,y_pred))
print('Cross validation score (cv=4) of Adaboost:',cross_val_score(model,X,y1,cv=4).mean())


#Random Forest
model=RandomForestRegressor(max_depth=3)
para_dict={'n_estimators':[20,50,80]}
clf=GridSearchCV(model,para_dict,cv=4,scoring='r2')
clf.fit(X,y1)
clf.best_params_
clf.best_score_

model=RandomForestRegressor(n_estimators=50, max_depth=3)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('MSE of Random Forest: ', mean_squared_error(y_test,y_pred))
print('Cross validation score (cv=4) of Random Forest:',cross_val_score(model,X,y1,cv=4).mean())

#SVR
para_dict={'gamma':[0.001,0.01,0.1,1],
            'C':[10,100,800,1000,1200,1500]}
model=GridSearchCV(SVR(kernel='rbf'),para_dict,cv=4,scoring='r2')
model.fit(X,y1)
model.best_params_
model.best_score_

model=SVR(kernel='rbf',C=1200,gamma=0.1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('MSE of SVR: ', mean_squared_error(y_test,y_pred))
print('Cross validation score (cv=4) of SVR:',cross_val_score(model,X,y1,cv=4).mean())

