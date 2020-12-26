#%%
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,f1_score,classification_report
import torch.nn.functional as F
import torch.utils.data as Data
import cv2
import os
import gdal
from lib import *

# %%
class dnn(torch.nn.Module):
    def __init__(self,n_feature,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_ouput):
        super(dnn,self).__init__()
        self.hidden1=torch.nn.Linear(n_feature,n_hidden1)
        self.hidden2=torch.nn.Linear(n_hidden1,n_hidden2)
        self.hidden3=torch.nn.Linear(n_hidden2,n_hidden3)
        self.hidden4=torch.nn.Linear(n_hidden3,n_hidden4)
        self.hidden5=torch.nn.Linear(n_hidden4,n_hidden5)
        self.out=torch.nn.Linear(n_hidden5,n_ouput)
    
    def forward(self,labeled_X_train):
        X_train=F.relu(self.hidden1(labeled_X_train))
        X_train=F.relu(self.hidden2(X_train))
        X_train=F.relu(self.hidden3(X_train))
        X_train=F.relu(self.hidden4(X_train))
        X_train=F.relu(self.hidden5(X_train))
        labeled_yPred_train=F.log_softmax(self.out(X_train))
        return labeled_yPred_train

class CoModel:
    def __init__(self,labeled_X_train1,labeled_yTrue_train1):
        self.net=dnn(n_feature=9,n_hidden1=32,n_hidden2=64,n_hidden3=96,n_hidden4=32,n_hidden5=16,n_ouput=3)
        self.labeled_X_train1=labeled_X_train1
        self.labeled_yTrue_train1=labeled_yTrue_train1


    def fit(self,epochs=20):
        train_dataset1 = Data.TensorDataset(self.labeled_X_train1,self.labeled_yTrue_train1)
        loader1 = Data.DataLoader(dataset=train_dataset1,batch_size=150,shuffle=True)
        optimizer1=torch.optim.SGD(self.net.parameters(),lr=0.02)
        for i in range(epochs):
            print(i)
            for step, (labeled_batch_X_train1,labeled_batch_yTrue_train1) in enumerate(loader1):
                labeled_batch_yPred_train1=self.net(labeled_batch_X_train1)
                loss1=torch.nn.CrossEntropyLoss(labeled_batch_yPred_train1,labeled_batch_yTrue_train1)
                print(step,loss1)
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()

# %%
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
label1=np.loadtxt(open(r"E:\project\images\sentinal\process\label1\train17.csv","r"),delimiter=',',skiprows=21,dtype=np.int)
label2=np.loadtxt(open(r"E:\project\images\sentinal\process\label1\train19.csv","r"),delimiter=',',skiprows=21,dtype=np.int)
loc1=label1[:,:2]
loc2=label2[:,:2]
train1=addY(loc1,data1total,[50,50,50])
train2=addY(loc2,data2total,[50,50,50])

# prepare mask
mask=cv2.imread(r"E:\project\images\sentinal\process\label1\mask.tif",0)
#unmask=cv2.imread(r"E:\project\images\sentinal\process\label\unmask.tif",0)
unmask=np.ones((mask.shape[0],mask.shape[1]))-mask
mask=mask.ravel()
unmask=unmask.ravel()
data1reshape=data1total.reshape((-1,data1total.shape[2]))
data2reshape=data2total.reshape((-1,data2total.shape[2]))

#初始化数据
features1=['blue1','grn1','red1','nir1','NDVI1','NDWI1','MSAVI1','MTVI1','VARI1']
features2=['blue2','grn2','red2','nir2','NDVI2','NDWI2','MSAVI2','MTVI2','VARI2']
data1df=pd.DataFrame(data1reshape,columns=features1)
data2df=pd.DataFrame(data2reshape,columns=features2)
data1df['change1']=unmask
data2df['change2']=unmask
data1df['row1']=np.array([i//772 for i in range(data1df.shape[0])])
data1df['col1']=np.array([i%772 for i in range(data1df.shape[0])])
data2df['row2']=np.array([i//772 for i in range(data2df.shape[0])])
data2df['col2']=np.array([i%772 for i in range(data2df.shape[0])])
data1df['prob1']=np.zeros((data1df.shape[0]))
data2df['prob2']=np.zeros((data2df.shape[0]))
data1df['guessClass1']=np.zeros((data1df.shape[0]))-1
data2df['guessClass2']=np.zeros((data2df.shape[0]))-1
data1df['acceptClass1']=np.zeros((data1df.shape[0]))-1
data2df['acceptClass2']=np.zeros((data2df.shape[0]))-1
datadf=pd.concat([data1df,data2df],axis=1)


# 准备数据
x1=train1[:,:-1]
y1=train1[:,-1]
x2=train2[:,:-1]
y2=train2[:,-1]

# load label
lab1=np.loadtxt(open(r"E:\project\images\sentinal\process\label1\test17.csv","r"),delimiter=',',skiprows=21,dtype=np.int)
lab2=np.loadtxt(open(r"E:\project\images\sentinal\process\label1\test19.csv","r"),delimiter=',',skiprows=21,dtype=np.int)
testloc1=lab1[:,:2]
testloc2=lab2[:,:2]
test1=addY(testloc1,data1total,[1000,1000,1000])
test2=addY(testloc2,data2total,[1000,1000,1000])
x_test1 = test1[:,:-1]
y_test1 = test1[:,-1]
x_test2 = test2[:,:-1]
y_test2 = test2[:,-1]


unlabeled_X_train1=torch.tensor(torch.from_numpy(datadf[datadf['change1']==0].loc[:,features1].values),dtype=torch.float32)
unlabeled_X_train2=torch.tensor(torch.from_numpy(datadf[datadf['change2']==0].loc[:,features2].values),dtype=torch.float32)
labeled_train1=torch.tensor(torch.from_numpy(train1),dtype=torch.float32)
labeled_train2=torch.tensor(torch.from_numpy(train2),dtype=torch.float32)
labeled_X_train1=torch.tensor(torch.from_numpy(x1),dtype=torch.float32)
labeled_X_train2=torch.tensor(torch.from_numpy(x2),dtype=torch.float32)
labeled_yTrue_train1=torch.tensor(torch.from_numpy(y1),dtype=torch.float32)
labeled_yTrue_train2=torch.tensor(torch.from_numpy(y2),dtype=torch.float32)
test1=torch.tensor(torch.from_numpy(test1),dtype=torch.float32)
test2=torch.tensor(torch.from_numpy(test2),dtype=torch.float32)
x_test1=torch.tensor(torch.from_numpy(x_test1),dtype=torch.float32)
x_test2=torch.tensor(torch.from_numpy(x_test2),dtype=torch.float32)
yTrue_test1=torch.tensor(torch.from_numpy(y_test1),dtype=torch.float32)
yTrue_test2=torch.tensor(torch.from_numpy(y_test2),dtype=torch.float32)


# %%
comodel=CoModel(labeled_X_train1,labeled_yTrue_train1)
comodel.fit()

# %%
