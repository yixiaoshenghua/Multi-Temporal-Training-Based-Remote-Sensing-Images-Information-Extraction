#%%
import gdal
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import joypy
from sklearn import svm
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.metrics import log_loss,confusion_matrix,plot_confusion_matrix,f1_score,classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,BaggingClassifier
import scipy.stats
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
    return ((data-data.min())/(dmax-dmin)*255).astype(np.int)


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


#%%
# 读取时间序列影像数据
n=5
image_path=r'G:\project\images\sentinal'
image1_name="170211.tif"
image2_name="170402.tif"
image3_name="171009.tif"
image4_name="171218.tif"
n=5
img1=getBands(os.path.join(image_path,image1_name))
img2=getBands(os.path.join(image_path,image2_name))
img3=getBands(os.path.join(image_path,image3_name))
img4=getBands(os.path.join(image_path,image4_name))
#%%
indexName=["NDVI","NDWI","MSAVI","MTVI","VARI"]
labdata1=returnLabeldata(img1,indexName,reshape=False,norm=True)
labdata2=returnLabeldata(img2,indexName,reshape=False,norm=True)
labdata3=returnLabeldata(img3,indexName,reshape=False,norm=True)
labdata4=returnLabeldata(img4,indexName,reshape=False,norm=True)
data1norm=maxminNorm(img1)
data2norm=maxminNorm(img2)
data3norm=maxminNorm(img3)
data4norm=maxminNorm(img4)
data1total=np.concatenate((data1norm,labdata1),axis=2)
data2total=np.concatenate((data2norm,labdata2),axis=2)
data3total=np.concatenate((data3norm,labdata3),axis=2)
data4total=np.concatenate((data4norm,labdata4),axis=2)
data1reshape=data1total.reshape((-1,data1total.shape[2]))
data2reshape=data2total.reshape((-1,data2total.shape[2]))
data3reshape=data3total.reshape((-1,data3total.shape[2]))
data4reshape=data4total.reshape((-1,data4total.shape[2]))





#%%
# # 构建数据df
# index=np.random.randint(low=0,high=img1.shape[0]*img1.shape[1],size=150)
# img1reshape=data1reshape[index,:4].astype(np.int)
# img2reshape=data2reshape[index,:4].astype(np.int)
# img3reshape=data3reshape[index,:4].astype(np.int)
# img4reshape=data4reshape[index,:4].astype(np.int)
# Classes=np.array([0]*img1reshape.shape[0]+[1]*img2reshape.shape[0]+[2]*img3reshape.shape[0]+[3]*img4reshape.shape[0]).reshape((-1,1))
# imgdf=pd.DataFrame(np.concatenate((np.concatenate((img1reshape,img2reshape,img3reshape,img4reshape),axis=0),Classes),axis=1),columns=['Blue','Green','Red','Nir','Class'])
# imgdf.loc[:,['Blue', 'Green','Red','Nir']]=imgdf.loc[:,['Blue', 'Green','Red','Nir']]
# imgdf.to_csv(r"imgdf.csv",index=False)
imgdf=pd.read_csv(r"imgdf.csv")
# 绘制各幅影像间数据分布差异
# %%
# Draw Plot
plt.figure(figsize=(16,10), dpi=200)
fig, axes = joypy.joyplot(imgdf,ylabelsize=15, labels=['T1','T2','T3','T4'], column=['Blue', 'Green','Red','Nir'], by="Class", ylim='own', figsize=(14,10))

# Decoration
#plt.title('Data distribution on different temporal remote sensing images', fontsize=22)
#plt.tight_layout()
plt.savefig(r'G:\project\ISPRS\images\temporal_difference.jpg')
plt.show()
# %%
imgdf=imgdf.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# %%
