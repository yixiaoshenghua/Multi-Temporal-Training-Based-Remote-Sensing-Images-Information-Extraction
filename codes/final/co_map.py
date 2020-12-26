#%%
import gdal
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn import svm
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.metrics import log_loss,confusion_matrix,plot_confusion_matrix,f1_score,classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,BaggingClassifier
import scipy.stats
from tqdm import tqdm 
# %%
dir=r'D:\map\17'
lst=[]
name=['2017.02.11','2017.04.02','2017.10.09','2017.12.18']
for j in range(1,5):
    maps=np.zeros((50,653,772),dtype=np.uint8)
    for i in range(50):
        map=cv2.imread(os.path.join(dir,'self_label_1unlabel_5c_100class'+str(j)+'_'+str(i)+'.tif'),0)
        maps[i,:,:]=map
    maps=maps.reshape((50,-1))
    lst.append(maps)
fig, ax = plt.subplots(2, 2,figsize=(20,15), sharex='col', sharey='row')
for j in range(2):
    for k in range(2):
        maps=lst[j*2+k]
        cnts=np.zeros((653*772,2))
        for i in range(653*772):
            cnts[i,0]=np.argmax(np.bincount(maps[:,i]))
            cnts[i,1]=np.max(np.bincount(maps[:,i]))/50
        hist=cnts[:,1].ravel()
        ax[j,k].hist(hist,facecolor="black", edgecolor="black", alpha=0.8)
        ax[j,k].tick_params(labelsize=15)
        ax[j,k].tick_params(labelsize=15)
        ax[j,k].set_ylim(0,250000)
        # if j==2 and k==1:
        #     ax[j,k].set_xlabel('consistency rate', fontsize=30)
        # if k==0 and j==1:
        #     ax[j,k].set_ylabel('pixel', fontsize=30)
        ax[j,k].set_title(name[j*2+k], fontsize=25)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
plt.savefig(r"G:\project\images\sentinal\process\output\plot\17\self_consistency_label1_class.jpg")

# %%
fig, ax = plt.subplots(3, 3,figsize=(20,15), sharex='col', sharey='row')
for j in range(3):
    for k in range(3):
        ax[j,k].hist(hist,facecolor="red", edgecolor="black", alpha=0.9)
        fontsize=20
        ax[j,k].tick_params(labelsize=15)
        ax[j,k].tick_params(labelsize=15)
        if j==2 and k==1:
            ax[j,k].set_xlabel('consistency rate', fontsize=fontsize)
        if k==0 and j==1:
            ax[j,k].set_ylabel('pixel', fontsize=fontsize)
        ax[j,k].set_title('size='+str((3*j+k)*5), fontsize=fontsize)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
# %%
epochs=np.arange(5,50,5,dtype=np.int)
plt.figure(figsize=(10,8))

# plt.plot(epochs,res[9:,0],color='green',marker='^',linestyle='-')
# plt.plot(epochs,res[9:,6],color='green',marker='o',linestyle='-')
plt.plot(epochs,res[9:,17],color='black',marker='^',linestyle='-')
plt.plot(epochs,res[9:,23],color='black',marker='o',linestyle='-')
# plt.plot(epochs,res[:9,0],color='green',marker='^',linestyle='-.')
# plt.plot(epochs,res[:9,6],color='green',marker='o',linestyle='-.')
plt.plot(epochs,res[:9,17],color='black',marker='^',linestyle='-.')
plt.plot(epochs,res[:9,23],color='black',marker='o',linestyle='-.')
plt.ylim(0.6,0.9)
plt.xlim(0,50)
plt.xticks(epochs)
plt.ylabel('F1-score',fontsize=20)
plt.xlabel("unlabeled dataSize",fontsize=20)
#plt.legend(['T1 (T1&T2) final','T2 (T1&T2) final','T1 final','T2 final','T1 (T1&T2) initial','T2 (T1&T2) initial','T1 initial','T2 initial'],fontsize=16)
plt.legend(['T1 final','T2 final','T1 initial','T2 initial'],fontsize=16)
#plt.legend(['T1 (T1&T2) final','T2 (T1&T2) final','T1 (T1&T2) initial','T2 (T1&T2) initial'],fontsize=16)
plt.title("F1-score of average (labeled dataSize="+str(i+1)+", c=1.0)",fontsize=20)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.savefig(r"G:\project\images\sentinal\process\output\plot\161103_170211\labelsize"+str(i+1)+"c_1_0_ave_self.jpg")
plt.show()

#%%
# %%
epochs=np.arange(1,21,dtype=np.int)
plt.figure(figsize=(10,8))
plt.plot(epochs,(res[0][:,18]+res[0][:,19])/2,color='black',marker='^',linestyle='-.')
plt.plot(epochs,(res[1][:,18]+res[1][:,19])/2,color='black',marker='*',linestyle='-.')
plt.plot(epochs,(res[2][:,18]+res[2][:,19])/2,color='black',marker='+',linestyle='-.')
plt.plot(epochs,(res[0][:,20]+res[0][:,21])/2,color='red',marker='^',linestyle='-.')
plt.plot(epochs,(res[1][:,20]+res[1][:,21])/2,color='red',marker='*',linestyle='-.')
plt.plot(epochs,(res[2][:,20]+res[2][:,21])/2,color='red',marker='+',linestyle='-.')
# plt.plot(epochs,res[3][:,17],color='black',marker='o',linestyle='-.')
# plt.plot(epochs,res[4][:,17],color='black',marker='v',linestyle='-.')
# plt.ylim(0.2,0.6)
plt.xlim(0,20)
plt.xticks(epochs)
plt.ylabel('Consistency Loss',fontsize=20)
plt.xlabel("epoch",fontsize=20)
plt.legend(['labeled data 1(co-train)','labeled data 2(co-train)','labeled data 3(co-train)','labeled data 1(single)','labeled data 2(single)','labeled data 3(single)'],fontsize=16)
plt.title("Consistency Loss between classifiers(unlabeled data 10 times c 1.0)",fontsize=20)
plt.tight_layout()
plt.savefig(r"E:\project\images\sentinal\process\output\label3unlabel10epoch20c1_0Coloss123.jpg")
plt.show()

# %%

def ave(df):
    res=np.zeros((2,24))
    for i in range(100):
        if i%2:
            res[0,:]+=df.iloc[i,:24].values
        else:
            res[1,:]+=df.iloc[i,:24].values
    res/=50
    return res

def genDataUnlabel(path,unlabelsize):
    res1=np.zeros((20,12))
    res2=np.zeros((20,12))
    for i in range(10):
        df=pd.read_csv(os.path.join(path,"label_{}unlabel_{}epochs_10sampletimes_50c_100.csv".format(2*i+1,unlabelsize)))
        res=ave(df)
        res1[i,:6]=res[0,:6]
        res1[10+i,:6]=res[1,:6]
        res2[i,:6]=res[0,6:12]
        res2[10+i,:6]=res[1,6:12]
        res1[i,6:]=res[0,12:18]
        res1[10+i,6:]=res[1,12:18]
        res2[i,6:]=res[0,18:24]
        res2[10+i,6:]=res[1,18:24]
    return res1,res2

def mave(df):
    res=np.zeros((2,48))
    for i in range(100):
        if i%2:
            res[0,:]+=df.iloc[i,:].values
        else:
            res[1,:]+=df.iloc[i,:].values
    res/=50
    return res

def mgenDataUnlabel(path,unlabelsize):
    res1=np.zeros((20,12))
    res2=np.zeros((20,12))
    res3=np.zeros((20,12))
    res4=np.zeros((20,12))
    for i in range(10):
        df=pd.read_csv(os.path.join(path,"multi_label_{}unlabel_{}epochs_10sampletimes_50c_100.csv".format(2*i+1,unlabelsize)))
        res=mave(df)
        res1[i,:6]=res[0,:6]
        res1[10+i,:6]=res[1,:6]
        res2[i,:6]=res[0,6:12]
        res2[10+i,:6]=res[1,6:12]
        res3[i,:6]=res[0,12:18]
        res3[10+i,:6]=res[1,12:18]
        res4[i,:6]=res[0,18:24]
        res4[10+i,:6]=res[1,18:24]
        res1[i,6:]=res[0,24:30]
        res1[10+i,6:]=res[1,24:30]
        res2[i,6:]=res[0,30:36]
        res2[10+i,6:]=res[1,30:36]
        res3[i,6:]=res[0,36:42]
        res3[10+i,6:]=res[1,36:42]
        res4[i,6:]=res[0,42:48]
        res4[10+i,6:]=res[1,42:48]
    return res1,res2,res3,res4

def suave(df):
    res=np.zeros((2,24))
    for i in range(100):
        if i%2:
            res[0,:]+=df.iloc[i,:].values
        else:
            res[1,:]+=df.iloc[i,:].values
    res/=50
    return res

def sugenDataUnlabel(path,unlabelsize):
    res1=np.zeros((20,6))
    res2=np.zeros((20,6))
    res3=np.zeros((20,6))
    res4=np.zeros((20,6))
    for i in range(10):
        df=pd.read_csv(os.path.join(path,'super_label{}_add{}.csv'.format((2*i+1)*5,unlabelsize)))
        res=suave(df)
        res1[i,:]=res[0,:6]
        res1[10+i,:]=res[1,:6]
        res2[i,:]=res[0,6:12]
        res2[10+i,:]=res[1,6:12]
        res3[i,:]=res[0,12:18]
        res3[10+i,:]=res[1,12:18]
        res4[i,:]=res[0,18:24]
        res4[10+i,:]=res[1,18:24]
    return res1,res2,res3,res4
# %%
res170211_1,res171009_1=genDataUnlabel(r"G:\project\images\sentinal\process\output\res_csv\170211_171009",5)
res170211_2,res171218_1=genDataUnlabel(r"G:\project\images\sentinal\process\output\res_csv\170211_171218",5)
res170402_1,res171009_2=genDataUnlabel(r"G:\project\images\sentinal\process\output\res_csv\170402_171009",5)
res170402_2,res171218_2=genDataUnlabel(r"G:\project\images\sentinal\process\output\res_csv\170402_171218",5)
res171009_3,res171218_3=genDataUnlabel(r"G:\project\images\sentinal\process\output\res_csv\171009_171218",5)
mres170211,mres170402,mres171009,mres171218=mgenDataUnlabel(r"G:\project\images\sentinal\process\output\res_csv\17",5)
sres170211,sres170402,sres171009,sres171218=sugenDataUnlabel(r"G:\project\images\sentinal\process\output\res_csv\super",50)
# %%
