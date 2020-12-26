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

# %%
dir=r'G:\project\images\sentinal\process\output\map\161103_170211'
lst=[]
for j in range(1,10):
    maps=np.zeros((20,653,772),dtype=np.uint8)
    for i in range(20):
        map=cv2.imread(os.path.join(dir,'self_label_1unlabel_'+str(j*5)+'class1_'+str(i)+'.tif'),0)
        maps[i,:,:]=map
    maps=maps.reshape((20,-1))
    lst.append(maps)
fig, ax = plt.subplots(3, 3,figsize=(20,15), sharex='col', sharey='row')
for j in range(3):
    for k in range(3):
        maps=lst[j*3+k]
        cnts=np.zeros((653*772,2))
        for i in range(653*772):
            cnts[i,0]=np.argmax(np.bincount(maps[:,i]))
            cnts[i,1]=np.max(np.bincount(maps[:,i]))/20
        hist=cnts[:,1].ravel()
        ax[j,k].hist(hist,facecolor="black", edgecolor="black", alpha=0.7)
        ax[j,k].tick_params(labelsize=15)
        ax[j,k].tick_params(labelsize=15)
        ax[j,k].set_ylim(0,250000)
        if j==2 and k==1:
            ax[j,k].set_xlabel('consistency rate', fontsize=30)
        if k==0 and j==1:
            ax[j,k].set_ylabel('pixel', fontsize=30)
        ax[j,k].set_title('size='+str((3*j+k)*5), fontsize=25)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
plt.savefig(r"G:\project\images\sentinal\process\output\plot\161103_170211\self_consistency_label1_class1.jpg")

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
