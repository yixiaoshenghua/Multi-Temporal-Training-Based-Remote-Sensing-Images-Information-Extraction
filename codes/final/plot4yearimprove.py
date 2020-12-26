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
#%%
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
#%%
data=np.zeros((110,288),dtype=np.float32)
i=0
for unlabel in range(5,55,5):
    for label in range(1,22,2):
        data[i,:]=csv2mat(r"G:\project\images\sentinal\process\output\res_csv\4\multi_label_{}unlabel_{}epochs_10sampletimes_50c_100.csv".format(label,unlabel))
        i+=1
df=pd.DataFrame(data,columns=cols)
for name in ['veg','shadow','water','road','building','total']:
    for train in ['multi','self']:
        df["{}_improved_{}_total".format(train,name)]=(df["{}_improved_{}_mean_1".format(train,name)]+df["{}_improved_{}_mean_2".format(train,name)]+df["{}_improved_{}_mean_3".format(train,name)]+df["{}_improved_{}_mean_4".format(train,name)])/4


# %%
#improve
for i,unlabel in enumerate(range(5,55,5)):
    labelsizes=list(range(1,22,2))
    plt.figure(figsize=(8,6), dpi= 200)
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'multi_improved_veg_mean_1'].values,color='green',marker='o')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'multi_improved_veg_mean_2'].values,color='green',marker='*')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'multi_improved_veg_mean_3'].values,color='green',marker='v')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'multi_improved_veg_mean_4'].values,color='green',marker='D')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'self_improved_veg_mean_1'].values,color='black',marker='o')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'self_improved_veg_mean_2'].values,color='black',marker='*')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'self_improved_veg_mean_3'].values,color='black',marker='v')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'self_improved_veg_mean_4'].values,color='black',marker='D')
    plt.axhline(y=0.0, c="gray", ls="-.", lw=2)
    xtick_location = labelsizes
    xtick_labels = labelsizes
    plt.xticks(ticks=xtick_location, labels=xtick_labels, fontsize=15,)
    plt.legend(['Multi-training T1','Multi-training T2','Multi-training T3','Multi-training T4',
                'Self-training T1','Self-training T2','Self-training T3','Self-training T4'],fontsize=15)
    plt.xlabel('Number of labeled samples for each class',fontsize=18)
    #plt.ylabel('F1-score',fontsize=14)
    #plt.ylim(0.5,1)
    plt.ylabel('F1-score',fontsize=18)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(r"G:\project\ISPRS\images\years4plot\multi_self_veg_unlabel{}_improved_trend_mean.jpg".format(unlabel))
    plt.show()
# %%
#final
for i,unlabel in enumerate(range(5,55,5)):
    labelsizes=list(range(1,22,2))
    plt.figure(figsize=(8,6), dpi= 200)
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'multi_final_veg_mean_1'].values,color='green',marker='o')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'multi_final_veg_mean_2'].values,color='green',marker='*')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'multi_final_veg_mean_3'].values,color='green',marker='v')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'multi_final_veg_mean_4'].values,color='green',marker='D')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'self_final_veg_mean_1'].values,color='black',marker='o')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'self_final_veg_mean_2'].values,color='black',marker='*')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'self_final_veg_mean_3'].values,color='black',marker='v')
    plt.plot(labelsizes,df.loc[i*11:i*11+10,'self_final_veg_mean_4'].values,color='black',marker='D')
    #plt.axhline(y=0.0, c="r", ls="-.", lw=2)
    xtick_location = labelsizes
    xtick_labels = labelsizes
    plt.xticks(ticks=xtick_location, labels=xtick_labels, fontsize=15,)
    plt.legend(['Multi-training T1','Multi-training T2','Multi-training T3','Multi-training T4',
                'Self-training T1','Self-training T2','Self-training T3','Self-training T4'],fontsize=15)
    plt.xlabel('Number of labeled samples for each class',fontsize=18)
    #plt.ylabel('F1-score',fontsize=14)
    #plt.ylim(0.5,1)
    plt.ylabel('F1-score',fontsize=18)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(r"G:\project\ISPRS\images\years4plot\multi_self_veg_unlabel{}_final_trend_mean.jpg".format(unlabel))
    plt.show()

