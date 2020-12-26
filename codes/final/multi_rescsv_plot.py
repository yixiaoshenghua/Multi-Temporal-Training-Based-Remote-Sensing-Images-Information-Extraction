#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
# %%
csv17dir=r'G:\project\images\sentinal\process\output\res_csv\4'
#csvsuperdir=r'G:\project\images\sentinal\process\output\res_csv\super'
dfs=[]
for i in range(2,21):
    df=pd.read_csv(os.path.join(csv17dir,"multi_label_1unlabel_5epochs_10sampletimes_50c_{}.csv".format(5*i)))
    #super_df=pd.read_csv(os.path.join(csvsuperdir,'super_label{}_add100.csv'.format(i*5)))
    #df=pd.concat([df,super_df],axis=1)
    dfs.append(df)
# %%
initial_dfs=[]
final_dfs=[]
for df in dfs:
    initial_df=df.loc[np.arange(0,50,2),:]
    final_df=df.loc[np.arange(1,50,2),:]
    initial_dfs.append(initial_df)
    final_dfs.append(final_df)
# %%
cols=[]
for time in ['init','final']:
    for stat in ['mean','std']:
        for train in ['multi','self']:
            for k in range(1,5):
                for name in ['veg','shadow','water','road','building','total']:            
                    cols.append(train+'_'+time+'_'+name+'_'+stat+'_'+str(k))
stat_df=pd.DataFrame(np.zeros((19,192)),columns=cols)
for i in range(19):
    mean_init=initial_dfs[i].values.mean(axis=0)
    std_init=initial_dfs[i].values.std(axis=0)
    mean_final=final_dfs[i].values.mean(axis=0)
    std_final=final_dfs[i].values.std(axis=0)
    stat_df.iloc[i,:]=np.concatenate((mean_init,std_init,mean_final,std_final))

#%%
stat_df['labelsize']=np.arange(0.1,1.05,0.05)
# %%
plt.figure(figsize=(8,6), dpi= 140)
labelsizes=[0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]
plt.plot(labelsizes,stat_df['multi_final_total_mean_1'],color='red',marker='o')
#plt.fill_between(labelsizes, stat_df['multi_final_veg_mean_1']-stat_df['multi_final_veg_std_1'], stat_df['multi_final_veg_mean_1']+stat_df['multi_final_veg_std_1'], color="#C1FFC1",alpha=0.3)  
#plt.plot(labelsizes,stat_df['multi_final_veg_mean_1']-stat_df['multi_init_veg_mean_1'],color='black',marker='o')
plt.plot(labelsizes,stat_df['self_final_total_mean_1'],color='green',marker='o')
#plt.fill_between(labelsizes, stat_df['self_final_veg_mean_1']-stat_df['self_final_veg_std_1'], stat_df['self_final_veg_mean_1']+stat_df['self_final_veg_std_1'],color="#C1FFC1",alpha=0.3)  
plt.plot(labelsizes,stat_df['multi_final_total_mean_4'],color='red',marker='D')
#plt.fill_between(labelsizes, stat_df['multi_final_veg_mean_3']-stat_df['multi_final_veg_std_3'], stat_df['multi_final_veg_mean_3']+stat_df['multi_final_veg_std_3'], color="#FFC1C1")  
#plt.plot(labelsizes,stat_df['multi_final_veg_mean_4']-stat_df['multi_init_veg_mean_4'],color='black',marker='D')
#plt.plot(labelsizes,stat_df['super_final_veg_mean_4']-stat_df['super_init_veg_mean_4'],color='#4A708B',marker='o')
#plt.fill_between(labelsizes, stat_df['multi_final_veg_mean_4']-stat_df['multi_final_veg_std_4'], stat_df['multi_final_veg_mean_4']+stat_df['multi_final_veg_std_4'], color="#AFEEEE",alpha=0.3) 
# plt.plot(labelsizes,stat_df['multi_final_veg_mean_3'],color='red',marker='*')
# #plt.fill_between(labelsizes, stat_df['multi_final_veg_mean_1']-stat_df['multi_final_veg_std_1'], stat_df['multi_final_veg_mean_1']+stat_df['multi_final_veg_std_1'], color="#FFC1C1")  
# plt.plot(labelsizes,stat_df['multi_init_veg_mean_3'],color='black',marker='*')
# #plt.fill_between(labelsizes, stat_df['multi_final_veg_mean_2']-stat_df['multi_final_veg_std_2'], stat_df['multi_final_veg_mean_2']+stat_df['multi_final_veg_std_2'], color="#FFC1C1")  
plt.plot(labelsizes,stat_df['self_final_total_mean_4'],color='green',marker='D')
#plt.fill_between(labelsizes, stat_df['self_final_veg_mean_4']-stat_df['self_final_veg_std_4'], stat_df['self_final_veg_mean_4']+stat_df['self_final_veg_std_4'], color="#AFEEEE",alpha=0.3)  
# plt.plot(labelsizes,stat_df['multi_init_veg_mean_4'],color='black',marker='v')
# #plt.fill_between(labelsizes, stat_df['multi_final_veg_mean_4']-stat_df['multi_final_veg_std_4'], stat_df['multi_final_veg_mean_4']+stat_df['multi_final_veg_std_4'], color="#FFC1C1")   
xtick_location = labelsizes
xtick_labels = labelsizes
plt.xticks(ticks=xtick_location, labels=xtick_labels, fontsize=12, alpha=.7,rotation=45)
plt.legend(['multi-training T1','self-training T1',
            'multi-training T2','self-training T2'],fontsize=14)
plt.xlabel('Threshold for each class',fontsize=14)
# plt.ylabel('Standard deviation',fontsize=14)
#plt.ylim(0.5,1)
plt.ylabel('F1-score of total final',fontsize=14)
plt.yticks(fontsize=12, alpha=.7)
plt.tight_layout()
#plt.savefig(r"G:\project\ISPRS\images\multi_self_veg_labelsize_improved_trend.jpg")
plt.show()
# %%
import matplotlib as mpl
plt.figure(figsize=(16,10), dpi= 80)
plt.plot('labelsize', 'self_final_veg_mean_1', data=stat_df, color='tab:green', label='self_final_veg_mean_1')
plt.fill_between(labelsizes,stat_df['self_final_veg_mean_1']-stat_df['self_final_veg_std_1'] , stat_df['self_final_veg_mean_1']+stat_df['self_final_veg_std_1'], color="#6F8F8D")
# Decoration
plt.ylim(0.7,1)
xtick_location = stat_df.labelsize.tolist()
xtick_labels = stat_df.labelsize.tolist()
plt.xticks(ticks=xtick_location, labels=xtick_labels, fontsize=12, alpha=.7)
plt.title("self_final_veg_mean (1 - 19)", fontsize=22)
plt.yticks(fontsize=12, alpha=.7)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)

plt.legend(loc='upper left')
plt.grid(axis='y', alpha=.3)
plt.show()
# %%
# %%
def gen(path,j,k,unlabelsize):
    dfs=[]
    for i in range(1,21,2):
        df=pd.read_csv(os.path.join(path,"label_{}unlabel_{}epochs_10sampletimes_50c_100.csv".format(i,unlabelsize)))
        dfs.append(df)
    initial_dfs=[]
    final_dfs=[]
    for df in dfs:
        initial_df=df.iloc[np.arange(0,50,2),:24]
        final_df=df.iloc[np.arange(1,50,2),:24]
        initial_dfs.append(initial_df)
        final_dfs.append(final_df)
    cols=[]
    for time in ['init','final']:
        for stat in ['mean','std']:
            for train in ['co','self']:
                for name in ['veg','shadow','water','road','building','total']:            
                    cols.append(train+'_'+time+'_'+name+'_'+stat+'_'+str(j))
                    cols.append(train+'_'+time+'_'+name+'_'+stat+'_'+str(k))
    s_df=pd.DataFrame(np.zeros((10,96)),columns=cols)
    for i in range(10):
        mean_init=initial_dfs[i].values.mean(axis=0)
        std_init=initial_dfs[i].values.std(axis=0)
        mean_final=final_dfs[i].values.mean(axis=0)
        std_final=final_dfs[i].values.std(axis=0)
        s_df.loc[i,cols]=np.concatenate((mean_init,std_init,mean_final,std_final))
    return s_df
# %%
df12=gen(r'G:\project\images\sentinal\process\output\res_csv\170211_170402',1,2,10)
df13=gen(r"G:\project\images\sentinal\process\output\res_csv\170211_171009",1,3,10)
df14=gen(r"G:\project\images\sentinal\process\output\res_csv\170211_171218",1,4,10)
df23=gen(r"G:\project\images\sentinal\process\output\res_csv\170402_171009",2,3,10)
df24=gen(r"G:\project\images\sentinal\process\output\res_csv\170402_171218",2,4,10)
df34=gen(r"G:\project\images\sentinal\process\output\res_csv\171009_171218",3,4,10)
#%%
for name in ['veg','shadow','water','road','building','total']:
    for time in ['init','final']:
        for stat in ['mean','std']:
            stat_df['co_{}_{}_{}_1'.format(time,name,stat)]=(df13['co_{}_{}_{}_1'.format(time,name,stat)]+df14['co_{}_{}_{}_1'.format(time,name,stat)]+df12['co_{}_{}_{}_1'.format(time,name,stat)])/3
            stat_df['co_{}_{}_{}_2'.format(time,name,stat)]=(df23['co_{}_{}_{}_2'.format(time,name,stat)]+df24['co_{}_{}_{}_2'.format(time,name,stat)]+df12['co_{}_{}_{}_2'.format(time,name,stat)])/3
            stat_df['co_{}_{}_{}_3'.format(time,name,stat)]=(df13['co_{}_{}_{}_3'.format(time,name,stat)]+df23['co_{}_{}_{}_3'.format(time,name,stat)]+df34['co_{}_{}_{}_3'.format(time,name,stat)])/3
            stat_df['co_{}_{}_{}_4'.format(time,name,stat)]=(df14['co_{}_{}_{}_4'.format(time,name,stat)]+df24['co_{}_{}_{}_4'.format(time,name,stat)]+df34['co_{}_{}_{}_4'.format(time,name,stat)])/3
#%%
tmp_df=pd.DataFrame((np.zeros((4,4))),columns=['self-learning','co-training','multi-training','super-training'])
tmp_df.loc[0,:]=stat_df.loc[0,['self_final_veg_mean_1','co_final_veg_mean_1','multi_final_veg_mean_1','self_final_veg_mean_1']].values
tmp_df.loc[1,:]=stat_df.loc[0,['self_final_veg_mean_2','co_final_veg_mean_2','multi_final_veg_mean_2','self_final_veg_mean_2']].values
tmp_df.loc[2,:]=stat_df.loc[0,['self_final_veg_mean_3','co_final_veg_mean_3','multi_final_veg_mean_3','self_final_veg_mean_3']].values
tmp_df.loc[3,:]=stat_df.loc[0,['self_final_veg_mean_4','co_final_veg_mean_4','multi_final_veg_mean_4','self_final_veg_mean_4']].values
#%%
plt.figure(figsize = (18, 15)) 
tmp_df.plot(kind='bar')
plt.ylim(0.7,1)
plt.legend(loc='upper left')
xtick_location=[0,1,2,3]
xtick_labels = ['02.11','04.02','10.09','12.18']
plt.xticks(ticks=xtick_location, labels=xtick_labels,rotation=45, fontsize=12, alpha=.7)
plt.ylabel('F1-score of vegetation')
plt.tight_layout()
plt.savefig(r"G:\project\ISPRS\images\final_compare_veg_1_10.jpg",dpi=150)
# %%
