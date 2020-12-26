#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
# %%
cols=[]
for time in ['init','final','improved']:
    for stat in ['mean','std']:
        for train in ['multi','self']:
            for k in range(1,5):
                for name in ['veg','shadow','water','road','building','total']:            
                    cols.append(train+'_'+time+'_'+name+'_'+stat+'_'+str(k))
def csv2mat(path):
    df=pd.read_csv(path)
    initial_df=df.loc[np.arange(0,50,2),:]
    final_df=df.loc[np.arange(1,50,2),:]
    improve_df=final_df.values - initial_df.values
    mean_init=initial_df.values.mean(axis=0)
    std_init=initial_df.values.std(axis=0)
    mean_final=final_df.values.mean(axis=0)
    std_final=final_df.values.std(axis=0)
    mean_improve=improve_df.mean(axis=0)
    std_improve=improve_df.std(axis=0)
    mat=np.concatenate((mean_init,std_init,mean_final,std_final,mean_improve,std_improve)).reshape((1,288))
    return mat
def maxminNorm(data):
    '''
    01标准化
    '''
    data=data.astype(np.float32)
    dmax=data.max()
    dmin=data.min()
    return (data-data.min())/(dmax-dmin)
#%%
data=np.zeros((110,288),dtype=np.float32)
i=0
for label in range(1,22,2):
    for unlabel in range(5,55,5):
        data[i,:]=csv2mat(r"G:\project\images\sentinal\process\output\res_csv\4\multi_label_{}unlabel_{}epochs_10sampletimes_50c_100.csv".format(label,unlabel))
        i+=1
df=pd.DataFrame(data,columns=cols)
df.index = pd.Series([str(label)+"_"+str(unlabel) for label in range(1,22,2) for unlabel in range(5,55,5)])
# %%
for name in ['veg','shadow','water','road','building','total']:
    for train in ['multi','self']:
        df["{}_improved_{}_mean".format(train,name)]=(df["{}_improved_{}_mean_1".format(train,name)]+df["{}_improved_{}_mean_2".format(train,name)]+df["{}_improved_{}_mean_3".format(train,name)]+df["{}_improved_{}_mean_4".format(train,name)])/4
        df["{}_improved_{}_std".format(train,name)]=(df["{}_improved_{}_std_1".format(train,name)]+df["{}_improved_{}_std_2".format(train,name)]+df["{}_improved_{}_std_3".format(train,name)]+df["{}_improved_{}_std_4".format(train,name)])/4
        df["{}_final_{}_mean".format(train,name)]=(df["{}_final_{}_mean_1".format(train,name)]+df["{}_final_{}_mean_2".format(train,name)]+df["{}_final_{}_mean_3".format(train,name)]+df["{}_final_{}_mean_4".format(train,name)])/4
        df["{}_final_{}_std".format(train,name)]=(df["{}_final_{}_std_1".format(train,name)]+df["{}_final_{}_std_2".format(train,name)]+df["{}_final_{}_std_3".format(train,name)]+df["{}_final_{}_std_4".format(train,name)])/4
# %%
plt.figure(figsize=(11,8))
improvedVeg_mat=df.multi_final_total_std.values
improvedVeg_mat=improvedVeg_mat.reshape((11,10))
improvedVeg_mat=improvedVeg_mat.T
ytick_labels=list(range(50,0,-5))
xtick_labels=list(range(1,22,2))
xtick_location=list(range(11))
ytick_location=list(range(10))
plt.imshow(improvedVeg_mat,cmap='OrRd_r')
plt.xticks(ticks=xtick_location, labels=xtick_labels,fontsize=15)
plt.yticks(ticks=ytick_location, labels=ytick_labels,fontsize=15)
plt.ylabel('Number of unlabeled samples',fontsize=20)
plt.xlabel('Number of labeled samples ',fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig(r"G:\project\ISPRS\images\labelUnlabel\labelunlabelplot_total_final_std.jpg",dpi=200)
# %%
