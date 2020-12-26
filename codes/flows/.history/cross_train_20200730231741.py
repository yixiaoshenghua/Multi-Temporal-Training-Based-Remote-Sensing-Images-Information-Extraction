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
y1=train1[:,-1:]
x2=train2[:,:-1]
y2=train2[:,-1:]

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

unlabeled_X_train1=datadf[datadf['change1']==0].loc[:,features1].values
unlabeled_X_train2=datadf[datadf['change2']==0].loc[:,features2].values

#%%

resdf=pd.DataFrame(np.zeros((5*20,16)),
                    columns=['17coveg','17cowater','17cobuilding','17cototal',
                            '19coveg','19cowater','19cobuilding','19cototal',
                            '17seveg','17sewater','17sebuilding','17setotal',
                            '19seveg','19sewater','19sebuilding','19setotal',])
#%%
for i in range(1,6):
    print(i)
    for j in range(20):
        ii=i*20+j
        x1_,y1_=selectSample(x1,y1,i)
        x2_,y2_=selectSample(x2,y2,i)
        tree1=GradientBoostedDecisionTree(n_iter=50,learning_rate=0.1)
        tree2=GradientBoostedDecisionTree(n_iter=50,learning_rate=0.1)
        for k in range(5):
            unlabeled_X_batch_train1,unlabeled_X_batch_train2=selectSample(unlabeled_X_train1,unlabeled_X_train2,i*10)
            tree1.fit(x1_,y1_.ravel().astype('int64'),x2_,y2_.ravel().astype('int64'),unlabeled_X_batch_train1,unlabeled_X_batch_train2)
            tree1.fit(x2_,y2_.ravel().astype('int64'),x1_,y1_.ravel().astype('int64'),unlabeled_X_batch_train2,unlabeled_X_batch_train1)
        y_pre1=tree1.predict(x_test1)
        y_pre2=tree2.predict(x_test2)
        res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
        res2=classification_report(y_test2,y_pre2,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
        resdf.iat[ii,0],resdf.iat[ii,1],resdf.iat[ii,2],resdf.iat[ii,3]=res1['veg']['f1-score'],res1['water']['f1-score'],res1['building']['f1-score'],res1['weighted avg']['f1-score']
        resdf.iat[ii,4],resdf.iat[ii,5],resdf.iat[ii,6],resdf.iat[ii,7]=res2['veg']['f1-score'],res2['water']['f1-score'],res2['building']['f1-score'],res2['weighted avg']['f1-score']
        print(res1['weighted avg']['f1-score'],res2['weighted avg']['f1-score'])
        tree3=GradientBoostedDecisionTree(n_iter=50,learning_rate=0.1)
        tree4=GradientBoostedDecisionTree(n_iter=50,learning_rate=0.1)
        tree3.fit(x1_,y1_.ravel().astype('int64'))
        tree4.fit(x2_,y2_.ravel().astype('int64'))
        y_pre3=tree3.predict(x_test1)
        y_pre4=tree4.predict(x_test2)
        res3=classification_report(y_test1,y_pre3,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
        res4=classification_report(y_test2,y_pre4,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
        resdf.iat[ii,8],resdf.iat[ii,9],resdf.iat[ii,10],resdf.iat[ii,11]=res3['veg']['f1-score'],res3['water']['f1-score'],res3['building']['f1-score'],res3['weighted avg']['f1-score']
        resdf.iat[ii,12],resdf.iat[ii,13],resdf.iat[ii,14],resdf.iat[ii,15]=res4['veg']['f1-score'],res4['water']['f1-score'],res4['building']['f1-score'],res4['weighted avg']['f1-score']
        print(res3['weighted avg']['f1-score'],res4['weighted avg']['f1-score'])
resdf.to_csv(r"E:\project\images\sentinal\process\output\res\resdf_consistency_GBDT_label12345_unlabel_10times_sample20.csv",index=None)
# %%

i=1
x1_,y1_=selectSample(x1,y1,i)
x2_,y2_=selectSample(x2,y2,i)


#%%
tree=GradientBoostedDecisionTree(n_iter=50,learning_rate=0.1)
for j in range(10):
    print(j)
    unlabeled_X_batch_train1,unlabeled_X_batch_train2=selectSample(unlabeled_X_train1,unlabeled_X_train2,50)
    tree.fit(x1_,y1_.ravel().astype('int64'),x2_,y2_.ravel().astype('int64'),unlabeled_X_batch_train1,unlabeled_X_batch_train2)
y_pre1=tree.predict(x_test1)
res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
print("consistency GBDT")
res1


# %%
tree=gbdt.GradientBoostedDecisionTree(n_iter=50,learning_rate=0.1)
tree.fit(x1_,y1_.ravel().astype('int64'))
y_pre1=tree.predict(x_test1)
res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
print("GBDT")
res1


# %%
