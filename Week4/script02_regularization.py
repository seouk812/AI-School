# libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# read data
df = pd.read_csv('data09_diabetes.csv')
df_data = df.iloc[:,:-1]
df_target = df['Y']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(
        df_data,df_target,test_size=0.33,random_state=0)

# linear regression
from sklearn.linear_model import LinearRegression
f = LinearRegression()
f.fit(xtrain,ytrain)
f.intercept_,f.coef_
f.score(xtrain,ytrain)
f.score(xtest,ytest)

# ridge regression
from sklearn.linear_model import Ridge
f = Ridge(alpha=0.5)
f.fit(xtrain,ytrain)
f.intercept_,f.coef_
f.score(xtrain,ytrain)
f.score(xtest,ytest)

# lasso regression
from sklearn.linear_model import Lasso
f = Lasso(alpha=1.0)
f.fit(xtrain,ytrain)
f.intercept_,f.coef_
f.score(xtrain,ytrain)
f.score(xtest,ytest)

# Elastic Net regression
from sklearn.linear_model import ElasticNet
f = ElasticNet(alpha=0.1,l1_ratio=0.5)
f.fit(xtrain,ytrain)
f.intercept_,f.coef_
f.score(xtrain,ytrain)
f.score(xtest,ytest)

# select parameter using cross-validation
from sklearn.model_selection import cross_val_score
exp = np.linspace(-2,0,21)
alphas = 10**exp
s = np.zeros((len(alphas),3))
for n in range(s.shape[0]):
    f = Lasso(alpha=alphas[n])
    f.fit(xtrain,ytrain)
    s[n,0] = f.score(xtrain,ytrain)
    s[n,1] = cross_val_score(f,xtrain,ytrain,cv=5).mean()
    s[n,2] = f.score(xtest,ytest)



plt.plot(exp,s[:,0],exp,s[:,1],exp,s[:,2],marker='o')
plt.legend(('Train','CV','Test'))
plt.show()

idx = np.argmax(s[:,1])
f = Ridge(alpha=alphas[idx])
f.fit(xtrain,ytrain)
f.coef_


# practice
df = pd.read_csv('data02_college.csv')
X = df.iloc[:,3:]
y = df['Accept']/df['Apps']
np.random.seed(1)
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.5,random_state=0)




















# PLEASE DO NOT GO DOWN BEFORE YOU TRY BY YOURSELF

###########################################################
# Practice Reference Code
###########################################################

# practice
df = pd.read_csv('data02_college.csv')
X = df.iloc[:,3:]
y = df['Accept']/df['Apps']
np.random.seed(1)
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.5,random_state=0)

# elastic net parameter search
exp = np.linspace(-4,1,31)
alphas = 10**exp
ratios = np.linspace(0,1,11)
s = np.zeros((len(alphas)*len(ratios),5))
idx = 0
for i in range(len(alphas)):
    for j in range(len(ratios)):
        f = ElasticNet(alpha=alphas[i],l1_ratio=ratios[j],max_iter=10000)
        f.fit(xtrain,ytrain)
        s[idx,0] = f.score(xtrain,ytrain)
        s[idx,1] = cross_val_score(f,xtrain,ytrain,cv=5).mean()
        s[idx,2] = f.score(xtest,ytest)      
        s[idx,3] = alphas[i]
        s[idx,4] = ratios[j]
        idx = idx+1
        
idx = np.argmax(s[:,1])        







