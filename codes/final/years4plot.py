#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
# %%
cols=[]
for time in ['init','final']:
    for stat in ['mean','std']:
        for train in ['multi','self']:
            for k in range(1,5):
                for name in ['veg','shadow','water','road','building','total']:            
                    cols.append(train+'_'+time+'_'+name+'_'+stat+'_'+str(k))
def csv2mat(path):
    df=pd.read_csv(path)
    initial_df=df.loc[np.arange(0,50,2),:]
    final_df=df.loc[np.arange(1,50,2),:]
    mean_init=initial_df.values.mean(axis=0)
    std_init=initial_df.values.std(axis=0)
    mean_final=final_df.values.mean(axis=0)
    std_final=final_df.values.std(axis=0)
    mat=np.concatenate((mean_init,std_init,mean_final,std_final)).reshape((1,192))
    return mat
# %%
label,unlabel=1,10
years_lst=["1678","1567","2678","1267","4678","3458","1234","2456","2346","2345"]
data=np.zeros((10,192),dtype=np.float32)
for i,years in enumerate(years_lst):
    data[i,:]=csv2mat(r"G:\project\images\sentinal\process\output\res_csv\year_diff_combine\\"+years+"\multi_label_{}unlabel_{}epochs_10sampletimes_50c_100.csv".format(label,unlabel))
df=pd.DataFrame(data,columns=cols)
df.index = pd.Series(["T"+str(t)+"*" for t in range(1,11)])

for i in range(1,5):
    for name in ['veg','shadow','water','road','building','total']:
        for train in ['multi','self']:
            df["{}_improved_{}_mean_{}".format(train,name,i)]=df["{}_final_{}_mean_{}".format(train,name,i)]-df["{}_init_{}_mean_{}".format(train,name,i)]

for name in ['veg','shadow','water','road','building','total']:
    for train in ['multi','self']:
        df["{}_improved_{}_total".format(train,name)]=(df["{}_improved_{}_mean_1".format(train,name)]+df["{}_improved_{}_mean_2".format(train,name)]+df["{}_improved_{}_mean_3".format(train,name)]+df["{}_improved_{}_mean_4".format(train,name)])/4

names=["multi_improved_total_total"]
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(111)
ax1.bar(df.index,df.multi_improved_veg_total)
ax1.set_ylabel(" F1-score",fontdict={'weight': 'normal', 'size': 15})

ax2 = ax1.twinx()  # this is the important function
ax2.plot(df['yearmonth'], df['total'], 'r')
ax2.set_ylabel('Difference of distribution',fontdict={'weight': 'normal', 'size': 15})
ax2.set_xlabel('Same')
plt.xlabel("Phases combination",fontsize=15)
plt.ylabel()
plt.tight_layout()
plt.savefig(r"G:\project\ISPRS\images\yearsdiff_compare_multiself_Totalimprove_label{}unlabel{}.jpg".format(label,unlabel),dpi=200)
# %%
