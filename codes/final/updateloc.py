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

def RGBnorm(data):
    data =data.astype(np.float32)
    dmax=data.max()
    dmin=data.min()
    res=(data-data.min())/(dmax-dmin)*255
    return res.astype(np.uint8)

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

#%%
k=4
image_path=r"G:\project\images\sentinal"
image_names=["170211.tif","170402.tif","171009.tif","171218.tif","180313.tif","180611.tif","180909.tif","190122.tif"]
label_names=["train170211.csv","train170402.csv","train171009.csv","train171218.csv","train180313.csv","train180611.csv","train180909.csv","train190122.csv"]
image_name_lst=image_names[:4]
label_path=r"G:\project\images\sentinal\process\labels"
label_name_lst=label_names[:4]
map_dir=r"G:\project\images\sentinal\process\output\map\4"
csv_path=r"G:\project\images\sentinal\process\output\res"
epochs=20
sampletimes=50
labelsize_min,labelsize_max,labelsize_step=1,1,2
unlabelsize_min,unlabelsize_max,unlabelsize_step=5,5,5
tradeoff_min,tradeoff_max,tradeoff_step=1,1,0.05
locpath=r"G:\project\images\sentinal\process\output\loc"
n=5
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
updatelocmat=np.loadtxt(open(r"G:\project\images\sentinal\process\output\loc\label1_unlabel5_sampletime0_updataloc.csv"),delimiter=',')
updatelocmat=updatelocmat.astype(np.int64)
# %%
left=118.772731583488
right=118.842081523422
top=32.095816954778
bottom=32.037156966725
#%%
locfile=np.zeros((100,10))
for i in range(5):
    locfile[:,2*i]=top-updatelocmat[:,i]//772*(top-bottom)/653
    locfile[:,2*i+1]=left+updatelocmat[:,i]%772*(right-left)/772
for i in range(4):
    locdf=pd.DataFrame(locfile[i*25:(i+1)*25,:],columns=['multi_x','multi_y','self1_x','self1_y','self2_x','self2_y','self3_x','self3_y','self4_x','self4_y'])
    locdf.to_csv(r"G:\project\images\sentinal\process\output\loc\label1_unlabel5_sample0_loc{}.csv".format(i+1),index=False)
# %%
loc_dir = r"G:\project\images\sentinal\process\output\loc"
locfile=np.zeros((5000,10))
for i in range(50):
    updatelocmat=np.loadtxt(open(r"G:\project\images\sentinal\process\output\loc\label1_unlabel5_sampletime{}_updataloc.csv".format(i)),delimiter=',')
    updatelocmat=updatelocmat.astype(np.int64)
    for j in range(5):
        for k in range(4):
            locfile[1250*k+i*25:1250*k+(i+1)*25,2*j]=top-updatelocmat[k*25:(k+1)*25,j]//772*(top-bottom)/653
            locfile[1250*k+i*25:1250*k+(i+1)*25,2*j+1]=left+updatelocmat[k*25:(k+1)*25,j]%772*(right-left)/772
# %%
for i in range(4):
    locdf=pd.DataFrame(locfile[i*1250:(i+1)*1250,:],columns=['multi_x','multi_y','self1_x','self1_y','self2_x','self2_y','self3_x','self3_y','self4_x','self4_y'])
    locdf.to_csv(r"G:\project\images\sentinal\process\output\loc\label1_unlabel5_sample_loc{}.csv".format(i+1),index=False)
# %%
