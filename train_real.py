import numpy as np
import sklearn.linear_model as sk
import sklearn.svm as svk
from sklearn.ensemble import RandomForestClassifier as rf
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics as sm

ltype=os.listdir("/home/pi/rpo/Audio-based-activity-recognition/data/test")
leat=os.listdir("/home/pi/rpo/Audio-based-activity-recognition/data/eat")
lhair=os.listdir("/home/pi/rpo/Audio-based-activity-recognition/data/hair")
llaund=os.listdir("/home/pi/rpo/Audio-based-activity-recognition/data/laund")
lvac=os.listdir("/home/pi/rpo/Audio-based-activity-recognition/data/vac")
l=[ltype,leat,lhair,llaund,lvac]
v=50


activ1=np.ndarray((5,v,1,39000))
for k in range(0,5):
    if k==2:
        v=35
    else:
        v=50
    for i in range(0,v):
        activ1[k][i]=np.loadtxt(l[k][i])
X=np.ndarray((5*v,39000))
y=np.ndarray(5*v)
m=0
for k in range(0,5):
    for i in range(0,v):
        X[m]=activ1[k][i]
        y[m]=k
        m=m+1

#clf1=svk.SVC()
#clf1.fit(X,y)

clf1=rf()
clf1.fit(X,y)
