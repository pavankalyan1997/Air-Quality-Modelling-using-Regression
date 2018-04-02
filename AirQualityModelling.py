import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

dataset=pd.read_csv('Copy of AirQualityUCI.csv');
X=dataset.iloc[:9357,6:16].values;
Y=dataset.iloc[:,16:19].values
Y1=dataset.iloc[:9357,16:17].values;
Y2=dataset.iloc[:9357,17:18].values;
Y3=dataset.iloc[:9357,18:19].values;

"""def FeatureScaling(X):
    for i in range(X.shape[1]):
        Min=min(X[:,i])
        Max=max(X[:,i])
        for j in range(X.shape[0]):
            X[j][i]=(X[j][i]-Min)/(Max-Min)
    return X     
            
"""

def MissingData(X):
    for i in range(X.shape[1]):
        avg=0
        count=0
        flag=0
        for j in range(X.shape[0]):
             if X[j][i]!=-200:
                 avg+=X[j][i]
                 flag=1
                 count+=1
        if flag==1:
            avg=(avg/count)
        for j in range(X.shape[0]):
             if X[j][i]==-200:
                 X[j][i]=avg

def PCA(X):
    cov_x=np.cov(X.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_x)
    eigen_pairs=[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)
    w=np.hstack((eigen_pairs[0][1][:,np.newaxis],
            eigen_pairs[1][1][:,np.newaxis],
            eigen_pairs[2][1][:,np.newaxis],
            eigen_pairs[3][1][:,np.newaxis],
            eigen_pairs[4][1][:,np.newaxis]))
    Xn=np.dot(X,w)
    return Xn



MissingData(X)
MissingData(Y1)
MissingData(Y2)
MissingData(Y3)
MissingData(Y)

"""X_scaled=FeatureScaling(X)
Y_scaled=FeatureScaling(X)
Y1_scaled=FeatureScaling(Y1)
Y2_scaled=FeatureScaling(Y2)
Y3_scaled=FeatureScaling(Y3)"""

#XPCA=PCA(X_scaled)

XPCA=PCA(X)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4)
XPCAPoly = poly_reg.fit_transform(XPCA)

division=int(X.shape[0]*0.95)

X_train=XPCAPoly[:division]
X_test=XPCAPoly[division:]

Y1_train=Y1[:division]
Y1_test=Y1[division:]

"""Y1_train=Y1_scaled[:division]
Y1_test=Y1_scaled[division:]"""

Y2_train=Y2[:division]
Y2_test=Y2[division:]

"""Y2_train=Y2_scaled[:division]
Y2_test=Y2_scaled[division:]"""

Y3_train=Y3[:division]
Y3_test=Y3[division:]

"""Y3_train=Y3_scaled[:division]
Y3_test=Y3_scaled[division:]"""

"""X_train=np.c_[np.ones(X_train.shape[0]),X_train]
X_test=np.c_[np.ones(X_test.shape[0]),X_test]"""

def LR(X,Y,X_test):
    W=np.dot(np.dot(inv(np.dot(X.T,X)+0.7*np.identity(np.dot(X.T,X).shape[0])),X.T),Y)
    Y_pred=np.dot(X_test,W)
    return Y_pred
    
y_pred1=LR(X_train,Y1_train,X_test)
y_pred2=LR(X_train,Y2_train,X_test)
y_pred3=LR(X_train,Y3_train,X_test)

e1=Y1_test-y_pred1
e2=Y2_test-y_pred2
e3=Y3_test-y_pred3

plt.scatter(Y1_test,y_pred1)
plt.plot(Y1_test,Y1_test)
plt.xlabel('Y1_test')
plt.ylabel('Y1_pred')
plt.show()

plt.scatter(Y2_test,y_pred2)
plt.plot(Y2_test,Y2_test)
plt.xlabel('Y2_test')
plt.ylabel('Y2_pred')
plt.show()

plt.scatter(Y3_test,y_pred3)
plt.plot(Y3_test,Y3_test)
plt.xlabel('Y3_test')
plt.ylabel('Y3_pred')
plt.show()


#implementation using library function
        
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_train, Y1_train)

plt.scatter(Y1_test,regressor1.predict(X_test))
plt.plot(Y1_test,Y1_test)
plt.ylabel('Y1_pred (ScikitLearn Library)')
plt.xlabel('Y1_test (ScikitLearn Library)')
plt.show()

regressor2 = LinearRegression()
regressor2.fit(X_train, Y2_train)

plt.scatter(Y2_test,regressor2.predict(X_test))
plt.plot(Y2_test,Y2_test)
plt.ylabel('Y2_pred (ScikitLearn Library)')
plt.xlabel('Y2_test (ScikitLearn Library)')
plt.show()

regressor3=LinearRegression()
regressor3.fit(X_train,Y3_train)

plt.scatter(Y3_test,regressor3.predict(X_test))
plt.plot(Y3_test,Y3_test)
plt.xlabel('Y3_test (ScikitLearn Library)')
plt.ylabel('Y3_pred (ScikitLearn Library)')
plt.show()







        
