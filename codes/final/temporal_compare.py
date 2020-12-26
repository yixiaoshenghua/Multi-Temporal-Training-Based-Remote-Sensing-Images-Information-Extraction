#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# %%
def csv2mat(path,k):
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
    mat=np.concatenate((mean_init,std_init,mean_final,std_final,mean_improve,std_improve)).reshape((1,k*72))
    return mat

#%%
colors={"vegfill":"lightgreen","veg":"green","totalfill":"lightcoral","total":"red"}
titles={"veg":"vegetation","total":"totally"}
for label in range(1,20,2):
    for unlabel in range(5,15,5):
        rescols=[]
        for name in ['veg','shadow','water','road','building','total']:
            for train in ['multi','self']:
                for time in ['improved','final']:
                    for stat in ['mean','std']:
                        rescols.append(train+'_'+ time+'_'+ name+'_'+ stat)
        resmat=np.zeros((6,48),dtype=np.float32)
        resdf=pd.DataFrame(resmat,columns=rescols)
        for k in range(3,9):
            cols=[]
            for time in ['init','final','improved']:
                for stat in ['mean','std']:
                    for train in ['multi','self']:
                        for n in range(1,k+1):
                            for name in ['veg','shadow','water','road','building','total']:            
                                cols.append(train+'_'+time+'_'+name+'_'+stat+'_'+str(n))
            data=np.zeros((1,k*72),dtype=np.float32)
            data=csv2mat(r"G:\project\images\sentinal\process\output\res_csv\{}\multi_label_{}unlabel_{}epochs_10sampletimes_50c_100.csv".format(k,label,unlabel),k)
            df=pd.DataFrame(data,columns=cols)
            for name in ['veg','shadow','water','road','building','total']:
                for train in ['multi','self']:
                    for time in ['improved','final']:
                        for stat in ['mean','std']:
                            tmp=0
                            for i in range(1,k+1):
                                tmp+=df["{}_{}_{}_{}_{}".format(train,time,name,stat,i)].values
                            resdf.loc[k-3,"{}_{}_{}_{}".format(train,time,name,stat)]=tmp/k
        for time in ["improved","final"]:
            for name in ["veg","total"]:
                plt.figure(figsize=(8,6),dpi=200)
                temp=np.arange(3,9)
                plt.plot(temp,resdf["multi_{}_{}_mean".format(time,name)].values,colors[name],marker='o',linewidth=3)
                plt.fill_between(temp, resdf["multi_{}_{}_mean".format(time,name)].values-resdf["multi_{}_{}_std".format(time,name)].values,resdf["multi_{}_{}_mean".format(time,name)].values+resdf["multi_{}_{}_std".format(time,name)].values, color=colors["{}fill".format(name)],alpha=0.3)  
                xtick_location = temp.tolist()
                xtick_labels = temp.tolist()
                plt.xticks(ticks=xtick_location, labels=xtick_labels, fontsize=18)
                plt.xlabel('Number of phases',fontsize=20)
                plt.ylabel('F1-score',fontsize=20)
                #plt.ylim(-0.3,0.3)
                plt.tight_layout()
                plt.savefig(r"G:\project\ISPRS\images\phasesNumber\numberOfPhasesCompare_label{}_unlabel{}_{}_{}_mean_fill.jpg".format(label,unlabel,time,name))
                

# %%
