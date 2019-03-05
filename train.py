#Program to store the training set and fitting the data 
import numpy as np
import sklearn.linear_model as sk
import sklearn.svm as svk
from sklearn.ensemble import RandomForestClassifier as rf
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics as sm

#Obtaining the training data from the individual text files
ltype=os.listdir("/home/pi/rpo/Audio-based-activity-recognition/data/feat1/test")
leat=os.listdir("/home/pi/rpo/Audio-based-activity-recognition/data/feat1/eat")
lhair=os.listdir("/home/pi/rpo/Audio-based-activity-recognition/data/feat1/hair")
llaund=os.listdir("/home/pi/rpo/Audio-based-activity-recognition/data/feat1/laund")
lvac=os.listdir("/home/pi/rpo/Audio-based-activity-recognition/data/feat1/vac")
l=[ltype,leat,lhair,llaund,lvac]
v=50

#Storing all the values in a single array
activ1=np.ndarray((5,v,1,13000))
for k in range(0,5):
    if k==2:
        v=35
    else:
        v=50
    for i in range(0,v):
        activ1[k][i]=np.loadtxt(l[k][i])
X=np.ndarray((5*v,13000))
y=np.ndarray(5*v)
m=0

#Storing the labels and the data to be fitted
for k in range(0,5):
    for i in range(0,v):
        X[m]=activ1[k][i]
        y[m]=k
        m=m+1

#Initialising random forest and storing
clf1=rf()
clf1.fit(X,y)
