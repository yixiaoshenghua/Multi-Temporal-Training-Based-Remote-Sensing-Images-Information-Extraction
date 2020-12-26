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
y_test1 = test1[:,-1]
x_test2 = test2[:,:-1]
y_test2 = test2[:,-1]

unchangelab=np.loadtxt(open(r'E:\project\images\sentinal\process\label2\un.csv','r'),delimiter=',',skiprows=29,dtype=np.int)
unchangeloc=unchangelab[:,:2]
unlabeled_train1=addY(unchangeloc,data1total,[100,100,100,100,120,8])
unlabeled_train2=addY(unchangeloc,data2total,[100,100,100,100,120,8])
unlabeled_X_train1=unlabeled_train1[:,:-1]
unlabeled_X_train2=unlabeled_train2[:,:-1]

#%%
resdf=pd.DataFrame(np.zeros((60,28)),
                    columns=['17cotrees','17coshadows','17cowater','17coroads','17cohouses','17cofallow','17cototal',
                            '19cotrees','19coshadows','19cowater','19coroads','19cohouses','19cofallow','19cototal',
                            '17setrees','17seshadows','17sewater','17seroads','17sehouses','17sefallow','17setotal',
                            '19setrees','19seshadows','19sewater','19seroads','19sehouses','19sefallow','19setotal',
                            ])
for i in range(1,6,2):
    print(i)
    for j in range(20):
        ii=(i-1)//2*20+j
        x1_,y1_=selectSample(x1,y1,i)
        x2_,y2_=selectSample(x2,y2,i)
        # cotrain
        tree1=GradientBoostedDecisionTree(n_iter=50,learning_rate=0.1)
        tree2=GradientBoostedDecisionTree(n_iter=50,learning_rate=0.1)
        for k in range(5):
            unlabeled_X_batch_train1,unlabeled_X_batch_train2=selectSample(unlabeled_X_train1,unlabeled_X_train2,i*10)
            tree1.fit(x1_,y1_.ravel().astype('int64'),x2_,y2_.ravel().astype('int64'),unlabeled_X_batch_train1,unlabeled_X_batch_train2)
            tree2.fit(x2_,y2_.ravel().astype('int64'),x1_,y1_.ravel().astype('int64'),unlabeled_X_batch_train2,unlabeled_X_batch_train1)
        y_pre1=tree1.predict(x_test1)
        y_pre2=tree2.predict(x_test2)
        res1=classification_report(y_test1,y_pre1,
                                    target_names=['trees','shadows','water','roads','houses','fallow'],
                                    output_dict=True)
        res2=classification_report(y_test2,y_pre2,
                                    target_names=['trees','shadows','water','roads','houses','fallow'],
                                    output_dict=True)
        resdf.iat[ii,0],resdf.iat[ii,1],resdf.iat[ii,2],resdf.iat[ii,3],resdf.iat[ii,4],resdf.iat[ii,5],resdf.iat[ii,6]=res1['trees']['f1-score'],res1['shadows']['f1-score'],res1['water']['f1-score'],res1['roads']['f1-score'],res1['houses']['f1-score'],res1['fallow']['f1-score'],res1['weighted avg']['f1-score']
        resdf.iat[ii,7],resdf.iat[ii,8],resdf.iat[ii,9],resdf.iat[ii,10],resdf.iat[ii,11],resdf.iat[ii,12],resdf.iat[ii,13]=res2['trees']['f1-score'],res2['shadows']['f1-score'],res2['water']['f1-score'],res2['roads']['f1-score'],res2['houses']['f1-score'],res2['fallow']['f1-score'],res2['weighted avg']['f1-score']
        print(res1['weighted avg']['f1-score'],res2['weighted avg']['f1-score'])
        # self-train
        tree3=gbdt.GradientBoostedDecisionTree(n_iter=50,learning_rate=0.1)
        tree4=gbdt.GradientBoostedDecisionTree(n_iter=50,learning_rate=0.1)
        tree3.fit(x1_,y1_.ravel().astype('int64'))
        tree4.fit(x2_,y2_.ravel().astype('int64'))
        y_pre3=tree3.predict(x_test1)
        y_pre4=tree4.predict(x_test2)
        res3=classification_report(y_test1,y_pre1,
                                    target_names=['trees','shadows','water','roads','houses','fallow'],
                                    output_dict=True)
        res4=classification_report(y_test2,y_pre2,
                                    target_names=['trees','shadows','water','roads','houses','fallow'],
                                    output_dict=True)
        resdf.iat[ii,14],resdf.iat[ii,15],resdf.iat[ii,16],resdf.iat[ii,17],resdf.iat[ii,18],resdf.iat[ii,19],resdf.iat[ii,20]=res3['trees']['f1-score'],res3['shadows']['f1-score'],res3['water']['f1-score'],res3['roads']['f1-score'],res3['houses']['f1-score'],res3['fallow']['f1-score'],res3['weighted avg']['f1-score']
        resdf.iat[ii,21],resdf.iat[ii,22],resdf.iat[ii,23],resdf.iat[ii,24],resdf.iat[ii,25],resdf.iat[ii,26],resdf.iat[ii,27]=res4['trees']['f1-score'],res4['shadows']['f1-score'],res4['water']['f1-score'],res4['roads']['f1-score'],res4['houses']['f1-score'],res4['fallow']['f1-score'],res4['weighted avg']['f1-score']
        print(res3['weighted avg']['f1-score'],res4['weighted avg']['f1-score'])
resdf.to_csv(r"E:\project\images\sentinal\process\output\res\resdf_consistency_GBDT_label135_unlabel_10times_sample20.csv",index=None)
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
res=np.loadtxt(open(r"E:\project\images\sentinal\process\output\res\resdf_consistency_gbdt_label123_unlabel10times.txt",'r'),delimiter=',',dtype=np.float32)

# %%
epochs=np.arange(1,21,dtype=np.int)
plt.figure(figsize=(10,8))
i=0
plt.plot(epochs,res[40:60,0],color='black',marker='^',linestyle='-')
plt.plot(epochs,res[40:60,1],color='black',marker='o',linestyle='-')
plt.plot(epochs,res[60:,0],color='black',marker='^',linestyle='-.')
plt.plot(epochs,res[60:,1],color='black',marker='o',linestyle='-.')
plt.ylim(0,1)
plt.xlim(0,20)
plt.xticks(epochs)
plt.ylabel('F1-score',fontsize=20)
plt.xlabel("epoch",fontsize=20)
plt.legend(['T1 (co-Loss)','T2 (co-Loss)','T1','T2'],fontsize=16)
plt.title("labeldata 2 unlabeldata 10",fontsize=20)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.savefig(r"E:\project\images\sentinal\process\output\GBDT_label2unlabel10Epoch20.jpg")
plt.show()

# %%
data1_std=np.std(data1total.reshape((-1,9)),axis=0,keepdims=True)
data2_std=np.std(data2total.reshape((-1,9)),axis=0,keepdims=True)
data1CSD=np.sum(((data1total.reshape((-1,9))-data2total.reshape((-1,9)))/data1_std)**2,axis=1,keepdims=True).reshape((653,772))
data2CSD=np.sum(((data1total.reshape((-1,9))-data2total.reshape((-1,9)))/data2_std)**2,axis=1,keepdims=True).reshape((653,772))
# %%
plt.hist(data1CSD.ravel(),bins=[200,250,300,350,400,450,500,550,600,650,700,750,800])

# %%
img1[:,:,:3][data1CSD>data1CSD.mean()]=np.array([0,0,0])
plt.imshow(maxminNorm(img1[:,:,:3]))
# %%
