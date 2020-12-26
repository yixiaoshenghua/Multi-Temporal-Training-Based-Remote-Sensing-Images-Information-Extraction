#%%
from lib import *
import gdal
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.metrics import log_loss,confusion_matrix,plot_confusion_matrix,f1_score,classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.semi_supervised import LabelPropagation
import scipy.stats
from trees.dt import *
from trees.rf import *
from trees.gbdt import *
from trees_ori import gbdt
features1=['blue1','grn1','red1','nir1','NDVI1','NDWI1','MSAVI1','MTVI1','VARI1']
features2=['blue2','grn2','red2','nir2','NDVI2','NDWI2','MSAVI2','MTVI2','VARI2']
#%%
# prepare data
path=r"E:\project\images\sentinal"
img1=getBands(os.path.join(path,"170211.tif"))
img2=getBands(os.path.join(path,"190919.tif"))
data1=img1
data2=img2
indexName=["NDVI","NDWI","MSAVI","MTVI","VARI"]
labdata1=returnLabeldata(data1,indexName,reshape=False,norm=True)
labdata2=returnLabeldata(data2,indexName,reshape=False,norm=True)
data1norm=maxminNorm(data1)
data2norm=maxminNorm(data2)
data1total=np.concatenate((data1norm,labdata1),axis=2)
data2total=np.concatenate((data2norm,labdata2),axis=2)

# load label
label1=np.loadtxt(open(r"E:\project\images\sentinal\process\label2\train17.csv","r"),delimiter=',',skiprows=29,dtype=np.int)
label2=np.loadtxt(open(r"E:\project\images\sentinal\process\label2\train19.csv","r"),delimiter=',',skiprows=29,dtype=np.int)
loc1=label1[:,:2]
loc2=label2[:,:2]
train1=addY(loc1,data1total,[20,20,20,20,20,20])
train2=addY(loc2,data2total,[20,20,20,20,20,20])



# 准备数据
x1=train1[:,:-1]
y1=train1[:,-1:]
x2=train2[:,:-1]
y2=train2[:,-1:]

# load label
lab1=np.loadtxt(open(r"E:\project\images\sentinal\process\label2\test17.csv","r"),delimiter=',',skiprows=29,dtype=np.int)
lab2=np.loadtxt(open(r"E:\project\images\sentinal\process\label2\test19.csv","r"),delimiter=',',skiprows=29,dtype=np.int)
testloc1=lab1[:,:2]
testloc2=lab2[:,:2]
test1=addY(testloc1,data1total,[1500,500,1500,500,500,500])
test2=addY(testloc2,data2total,[1500,500,1500,500,500,500])
x_test1 = test1[:,:-1]
y_test1 = test1[:,-1:]
x_test2 = test2[:,:-1]
y_test2 = test2[:,-1:]

unchangelab=np.loadtxt(open(r'E:\project\images\sentinal\process\label2\un.csv','r'),delimiter=',',skiprows=29,dtype=np.int)
unchangeloc=unchangelab[:,:2]
unlabeled_train1=addY(unchangeloc,data1total,[100,100,100,100,120,8])
unlabeled_train2=addY(unchangeloc,data2total,[100,100,100,100,120,8])
unlabeled_X_train1=unlabeled_train1[:,:-1]
unlabeled_X_train2=unlabeled_train2[:,:-1]



# %%
import numpy as np
import sklearn   
import scipy.io as scio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class TriTraining:
    def __init__(self, classifier):
        if sklearn.base.is_classifier(classifier):
            self.classifiers = [sklearn.base.clone(classifier) for i in range(3)]
        else:
            self.classifiers = [sklearn.base.clone(classifier[i]) for i in range(3)]
            
    def fit(self, L_X, L_y, U_X):
        for i in range(3):
            sample = sklearn.utils.resample(L_X, L_y)  
            self.classifiers[i].fit(*sample)   
        e_prime = [0.5]*3
        l_prime = [0]*3
        e = [0]*3
        update = [False]*3
        Li_X, Li_y = [[]]*3, [[]]*3#to save proxy labeled data
        improve = True
        self.iter = 0
        
        while improve:
            self.iter += 1#count iterations 
            
            for i in range(3):    
                j, k = np.delete(np.array([0,1,2]),i)
                update[i] = False
                e[i] = self.measure_error(L_X, L_y, j, k)
                if e[i] < e_prime[i]:
                    U_y_j = self.classifiers[j].predict(U_X)
                    U_y_k = self.classifiers[k].predict(U_X)
                    Li_X[i] = U_X[U_y_j == U_y_k]#when two models agree on the label, save it
                    Li_y[i] = U_y_j[U_y_j == U_y_k]
                    if l_prime[i] == 0:#no updated before
                        l_prime[i]  = int(e[i]/(e_prime[i] - e[i]) + 1)
                    if l_prime[i] < len(Li_y[i]):
                        if e[i]*len(Li_y[i])<e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i]/(e_prime[i] - e[i]):
                            L_index = np.random.choice(len(Li_y[i]), int(e_prime[i] * l_prime[i]/e[i] -1))
                            Li_X[i], Li_y[i] = Li_X[i][L_index], Li_y[i][L_index]
                            update[i] = True
             
            for i in range(3):
                if update[i]:
                    self.classifiers[i].fit(np.append(L_X,Li_X[i],axis=0), np.append(L_y, Li_y[i], axis=0))
                    e_prime[i] = e[i]
                    l_prime[i] = len(Li_y[i])
    
            if update == [False]*3:
                improve = False#if no classifier was updated, no improvement


    def predict(self, X):
        pred = np.asarray([self.classifiers[i].predict(X) for i in range(3)])
        pred[0][pred[1]==pred[2]] = pred[1][pred[1]==pred[2]]
        return pred[0]
        
    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.predict(X))
        
    def measure_error(self, X, y, j, k):
        j_pred = self.classifiers[j].predict(X)
        k_pred = self.classifiers[k].predict(X)
        wrong_index =np.logical_and(j_pred != y, k_pred==j_pred)
        #wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
        return sum(wrong_index)/sum(j_pred == k_pred)


# traindata  = x1
# trainlabel = y1
# testdata   = x_test1
# testlabel  = y_test1

# data = np.row_stack([traindata,testdata])
# label = np.row_stack([trainlabel,testlabel]).argmax(axis=1)

# train_index = np.random.choice(data.shape[0],200,replace=False)
# rest_index = list(set(np.arange(data.shape[0])) - set(train_index))
# test_index = np.random.choice(rest_index,500,replace=False)
# u_index = list(set(rest_index) - set(test_index))

# traindata = data[train_index]
# trainlabel = label[train_index]

def selectSample(x,y,sample):
    m=np.zeros((sample*6),dtype=np.int)
    for i in range(6):
        m[i*sample:(i+1)*sample]=np.random.randint(i*20,(i+1)*20,size=sample)
    return x[m,:],y[m,:]


np.random.shuffle(test1)

traindata,trainlabel=selectSample(x1,y1,sample=5)
trainlabel=trainlabel.ravel()
size=200
testdata = test1[:size,:9]
testlabel = test1[:size,-1]

udata = test1[size:1000,:9]

print(traindata.shape,testdata.shape,udata.shape)
print(trainlabel.shape,testlabel.shape)

clf = RandomForestClassifier()
clf.fit(traindata,trainlabel)
res1 = clf.predict(testdata)
print(accuracy_score(res1,testlabel))


TT = TriTraining([RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier()])
TT.fit(traindata,trainlabel,udata)
res2 = TT.predict(testdata)
print(accuracy_score(res2,testlabel))

# %%
