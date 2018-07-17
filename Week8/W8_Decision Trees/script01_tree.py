# libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


###########################################################
# Classification Tree
###########################################################

# read data from file
df = pd.read_csv('data01_iris.csv')
train_idx = list(np.arange(0,30))+list(np.arange(50,80))+list(np.arange(100,130))
test_idx = list(set(np.arange(0,150)).difference(train_idx))
X = df.iloc[:,:-1]
Y = df['Species'].factorize()[0]
xtrain, ytrain = X.iloc[train_idx,:], Y[train_idx]
xtest, ytest = X.iloc[test_idx,:], Y[test_idx]

# classification tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(xtrain,ytrain)
yhat_train = clf.predict(xtrain)
yhat_train_prob = clf.predict_proba(xtrain)
yhat_test = clf.predict(xtest)
yhat_test_prob = clf.predict_proba(xtest)
clf.score(xtrain,ytrain)
clf.score(xtest,ytest)

# lda for comparison
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(xtrain,ytrain)
lda.score(xtrain,ytrain)
lda.score(xtest,ytest)

# make data dirty
np.random.seed(0)
xtrain2 = xtrain + 2*np.random.randn(np.prod(xtrain.shape)).reshape(90,4)
xtest2 = xtest + 2*np.random.randn(np.prod(xtest.shape)).reshape(60,4)

# classification tree
clf = DecisionTreeClassifier(max_leaf_nodes=5)
clf.fit(xtrain2,ytrain)
clf.score(xtrain2,ytrain)
clf.score(xtest2,ytest)


###########################################################
# Regression Tree
###########################################################

# read data
df = pd.read_csv('data05_boston.csv')
X, Y = df.iloc[:,:-1], df['medv']
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.33,random_state=0)

# regression tree
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(max_leaf_nodes=22)
clf.fit(xtrain,ytrain)
clf.score(xtrain,ytrain)
clf.score(xtest,ytest)

# linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(xtrain,ytrain)
lm.score(xtrain,ytrain)
lm.score(xtest,ytest)


###########################################################
# Practice
###########################################################







































# PLEASE DO NOT GO DOWN BEFORE YOU TRY BY YOURSELF

###########################################################
# Practice Reference Code
###########################################################

# read data
df = pd.read_csv('data05_boston.csv')
X, Y = df.iloc[:,:-1], df['medv']
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.33,random_state=0)
np.random.seed(0)

# cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
s = np.zeros((50,3))
for n in range(s.shape[0]):
    f = DecisionTreeRegressor(max_leaf_nodes=n+2,random_state=None)
    f.fit(xtrain,ytrain)
    s[n,0] = f.score(xtrain,ytrain)
    s[n,1] = cross_val_score(f,xtrain,ytrain,cv=5).mean()
    s[n,2] = f.score(xtest,ytest)

plt.plot(s,marker='o')
plt.legend(('Train','CV','Test'))
plt.show()

from scipy.interpolate import UnivariateSpline
spl = UnivariateSpline(np.arange(s.shape[0]),s[:,1],s=0.1)
s2 = spl(np.arange(s.shape[0]))

plt.plot(s,marker='o')
plt.plot(s2,'r',linewidth=2)
plt.legend(('Train','CV','Test'))
plt.show()

idx = np.argmax(s2)
s[idx,:]


