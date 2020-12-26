#%%
import gdal
import os
import numpy as np
import pandas as pd
import time
import cv2
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
# %%
features1=['blue1','grn1','red1','nir1','NDVI1','NDWI1','MSAVI1','MTVI1','VARI1']
features2=['blue2','grn2','red2','nir2','NDVI2','NDWI2','MSAVI2','MTVI2','VARI2']

def getBands(filename):
    img = gdal.Open(filename)
    bands = []
    for i in range(img.RasterCount):
        band = img.GetRasterBand(i+1).ReadAsArray().astype(np.float)
        bands.append(band)
    bands = np.dstack((bands[0],bands[1],bands[2],bands[3]))
    return bands

def maxminNorm(data):
    '''
    01标准化
    '''
    data=data.astype(np.float32)
    dmax=data.max()
    dmin=data.min()
    return (data-data.min())/(dmax-dmin)

def index(data,ind="NDVI"):
    '''
    input:原始数据，默认保持影像形状，默认计算NDVI
    returns:相应指数
    '''
    nir=data[:,:,3]
    red=data[:,:,2]
    grn=data[:,:,1]
    blue=data[:,:,0]
    if ind=="NDVI":# 归一化植被指数 
        NDVI=(nir-red)/(nir+red+1e-8)
        return NDVI
    elif ind=="NDWI":# 归一化水体指数
        NDWI=(grn-nir)/(grn+nir+1e-8)
        return NDWI
    elif ind=="CIg":# 叶绿素指数-绿边 
        CIg=nir/(grn+1e-8)-1
        return CIg
    elif ind=="EVI":# 增强型植被指数 
        EVI=2.5*(nir-red)/(nir+ 6*red-7.5*blue+1+1e-8)
        return EVI
    elif ind=="GNDVI":# 绿光归一化差值植被指数
        GNDVI = (nir - grn)/(nir + grn+1e-8)
        return GNDVI
    elif ind=="MSAVI":# 修正土壤调节植被指数
        MSAVI=0.5*(2*(nir +1)-np.sqrt((2*nir+1)**2-8*(nir-red)))
        return MSAVI
    elif ind=="MTVI":# 修正型三角植被指数
        MTVI=1.5*(1.2*(nir-grn)-2.5*(red-grn))/np.sqrt((2*nir+1)**2-(6*nir-5*np.sqrt(red))-0.5)
        return MTVI
    elif ind=="SAVI":# 土壤调节植被指数
        L=0.5
        SAVI=((nir-red)/(nir+red+L))*(1+L)
        return SAVI
    elif ind=="VARI":# 可视化大气阻抗指数
        VARI=(grn-red)/(grn+red-blue+1e-8)
        return VARI
    
def returnLabeldata(data,indexName,df=False,reshape=True,norm=False):
    h,w,d=data.shape
    dataIdx=np.zeros((data.shape[0],data.shape[1],len(indexName)),dtype=np.float32)
    for i,idx in enumerate(indexName):
        if norm==True:
            dataIdx[:,:,i]=maxminNorm(index(data,ind=idx))
        else:
            dataIdx[:,:,i]=index(data,ind=idx)
    if len(np.unique(np.argwhere(np.isnan(dataIdx))[:,2]))!=0:
        print(np.unique(np.argwhere(np.isnan(dataIdx))[:,2]))
    if df==True:
        res=pd.DataFrame(dataIdx.reshape((h*w,-1)),columns=indexName)
    if reshape==True:
        res=dataIdx.reshape((h*w,-1))
    else:
        res=dataIdx
    return res

def addY(loc,data,nums):#data3维
    #loc=loc.astype(np.int)
    y=np.array([0]*nums[0]+[1]*nums[1]+[2]*nums[2]+[3]*nums[3]+[4]*nums[4]).reshape((sum(nums),1))
    x=np.zeros((sum(nums),data.shape[2]))
    for i,yx in enumerate(loc):
        x[i,:]=data[yx[1],yx[0],:]
    train=np.concatenate((x,y),axis=1)
    return train

def returnUnlabeledData(data1,data2):
    unmask=np.ones((data1.shape[0]),dtype=np.int)
    diff_map=data1-data2
    diff_std=np.std(diff_map,axis=0)
    csd=np.sum(diff_map/diff_std,axis=1)
    unchanged_pos=np.where(csd<csd.mean())[0]
    unmask[unchanged_pos]=0
    return unmask

def random_train_test_split(data,size,train_rate=0.2,test_rate=0.8):
    n=np.unique(data[:,-1]).shape[0]
    train_class_size=int(size*train_rate)
    test_class_size=int(size*test_rate)
    X_train=np.zeros((n*train_class_size,data.shape[1]-1))
    X_test=np.zeros((n*test_class_size,data.shape[1]-1))
    y_train=np.zeros((n*train_class_size))
    y_test=np.zeros((n*test_class_size))
    for i in range(n):
        tmp=data[i*size:(i+1)*size,:]
        np.random.shuffle(tmp)
        X_train[i*train_class_size:(i+1)*train_class_size,:],X_test[i*test_class_size:(i+1)*test_class_size,:],y_train[i*train_class_size:(i+1)*train_class_size],y_test[i*test_class_size:(i+1)*test_class_size] = train_test_split(tmp[:,:-1],tmp[:,-1:].ravel(),test_size=test_rate,random_state=1023)
    return X_train, X_test, y_train.reshape((-1,1)), y_test.reshape((-1,1))

def selectSample(x,y,sample):
    m=np.zeros((sample*n),dtype=np.int)
    for i in range(n):
        m[i*sample:(i+1)*sample]=np.random.randint(i*200,(i+1)*200,size=sample)
    return x[m,:],y[m,:]

def fk(plst):
    m,s=1,0
    n=len(plst)
    for p in plst:
        m*=p
        s+=p
    return (m**(2/n))/(s/n+1e-8)

def report(test,models):
    y_preds=[]
    ress=[]
    for i in range(k):
        y_preds.append(models[i].predict(test[2*i]))
        ress.append(classification_report(test[2*i+1],y_preds[i],target_names=['veg','shadow','water','road','building'],output_dict=True))
    result=np.zeros((k,6))
    for i in range(k):
        res=ress[i]
        result[i,:]=np.array([res['veg']['f1-score'],res['shadow']['f1-score'],res['water']['f1-score'],res['road']['f1-score'],res['building']['f1-score'],res['weighted avg']['f1-score']])
    return result

def compute_diff4(datas):
    '''
    data: h*w*n
    '''
    n,k = datas[0].shape
    std_data=np.zeros((n,k),dtype=np.float32)
    dataStack=np.zeros((n,k,4),dtype=np.float32)
    for i in range(4):
        dataStack[:,:,i]=datas[i]
    std_data=np.std(dataStack,axis=2)
    diff=np.sum(std_data**2)
    return diff


#%%
k=8
image_path=r"G:\project\images\sentinal"
image_name_lst=["170211.tif","170402.tif","171009.tif","171218.tif","180313.tif","180611.tif","180909.tif","190122.tif"]
imgs=[]
indexName=["NDVI","NDWI","MSAVI","MTVI","VARI"]
labdatas=[]
datanorms=[]
datatotals=[]
datareshapes=[]
for i in range(k):
    imgs.append(getBands(os.path.join(image_path,image_name_lst[i])))
    labdatas.append(returnLabeldata(imgs[i],indexName,reshape=False,norm=True))
    datanorms.append(maxminNorm(imgs[i]))
    datatotals.append(np.concatenate((datanorms[i],labdatas[i]),axis=2))
    datareshapes.append(datatotals[i].reshape((-1,datatotals[i].shape[2])))
# %%
year_diff_dict={}
year_diff_lst=[]
for i in range(8):
    for j in range(i+1,8):
        for k in range(j+1,8):
            for m in range(k+1,8):
                diff=compute_diff4([datareshapes[i],datareshapes[j],datareshapes[k],datareshapes[m]])
                year_diff_lst.append(['T{}&T{}&T{}&T{}'.format(i+1,j+1,k+1,m+1),diff])
year_diff_mat=np.array(year_diff_lst)
#%%
year_diff_mat[:,1]=maxminNorm(year_diff_mat[:,1])
year_diff_mat=year_diff_mat[year_diff_mat[:,1].argsort()]
# %%
plt.rcParams["axes.labelsize"] = 25
plt.figure(figsize=(15, 40),dpi=200)
diff_df=pd.DataFrame(year_diff_mat,columns=['combination of years','data distribution difference'])
sns.barplot(x='data distribution difference', y='combination of years', data=diff_df)
plt.xticks(rotation=30) 
xtick_location=[i for i in range(70)]
plt.tick_params(labelsize=20)
#plt.xticks(ticks=xtick_location,rotation=45, fontsize=20)
plt.tight_layout()
plt.savefig(r"G:\project\ISPRS\images\yearsCombineDiff_std.jpg")
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
df.index = pd.Series(["C"+str(t) for t in range(1,11)])
for i in range(1,5):
    for name in ['veg','shadow','water','road','building','total']:
        for train in ['multi','self']:
            df["{}_improved_{}_mean_{}".format(train,name,i)]=df["{}_final_{}_mean_{}".format(train,name,i)]-df["{}_init_{}_mean_{}".format(train,name,i)]

for name in ['veg','shadow','water','road','building','total']:
    for train in ['multi','self']:
        df["{}_improved_{}_total".format(train,name)]=(df["{}_improved_{}_mean_1".format(train,name)]+df["{}_improved_{}_mean_2".format(train,name)]+df["{}_improved_{}_mean_3".format(train,name)]+df["{}_improved_{}_mean_4".format(train,name)])/4

yeardiffmat = np.concatenate((year_diff_mat[:5,1],year_diff_mat[-5:,1])).astype(np.float32)
df['difference'] = yeardiffmat[::-1]
#%%
names=["multi_improved_veg_total"]
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(111)
ax1.bar(df.index,df.multi_improved_total_total,color='lightcoral')
ax1.set_ylabel(" F1-score",fontdict={'weight': 'normal', 'size': 20})
ax1.tick_params(labelsize=18)

ax2 = ax1.twinx()  # this is the important function
ax2.plot(df.index, df.difference, color='gray',marker='o',linewidth=3)
ax2.set_ylabel('Difference of distribution',fontdict={'weight': 'normal', 'size': 20})
ax2.set_xlabel('Same')
plt.xlabel("Phases combination",fontsize=20)
ax2.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig(r"G:\project\ISPRS\images\phasesCombine\yearsdiff_compare_multiself_totalimprove_label{}unlabel{}.jpg".format(label,unlabel),dpi=200)
# %%
