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

def f4(a,b,c,d):
    return (a*b*c*d)**(1/2)/((a+b+c+d)/4+1e-8)

def report(test,model1,model2,model3,model4):
    x_test1,y_test1,x_test2,y_test2,x_test3,y_test3,x_test4,y_test4=test
    y1_pred=model1.predict(x_test1)
    y2_pred=model2.predict(x_test2)
    y3_pred=model3.predict(x_test3)
    y4_pred=model4.predict(x_test4)
    res1=classification_report(y_test1,y1_pred,target_names=['veg','shadow','water','road','building'],output_dict=True)
    res2=classification_report(y_test2,y2_pred,target_names=['veg','shadow','water','road','building'],output_dict=True)
    res3=classification_report(y_test3,y3_pred,target_names=['veg','shadow','water','road','building'],output_dict=True)
    res4=classification_report(y_test4,y4_pred,target_names=['veg','shadow','water','road','building'],output_dict=True)
    ress=[res1,res2,res3,res4]
    result=np.zeros((4,6))
    for i in range(4):
        res=ress[i]
        result[i,:]=np.array([res['veg']['f1-score'],res['shadow']['f1-score'],res['water']['f1-score'],res['road']['f1-score'],res['building']['f1-score'],res['weighted avg']['f1-score']])
    return result

def superrun(train,test,data,labelsize=50,unlabelsize=5,epochs=10,sampletimes=50):
    resdf=pd.DataFrame(np.zeros((sampletimes*2,24)),
                    columns=['super_veg1','super_shadow1','super_water1','super_road1','super_building1','super_total1',
                            'super_veg2','super_shadow2','super_water2','super_road2','super_building2','super_total2',
                            'super_veg3','super_shadow3','super_water3','super_road3','super_building3','super_total3',
                            'super_veg4','super_shadow4','super_water4','super_road4','super_building4','super_total4',
                            ])
    for sampletime in tqdm(range(sampletimes)):
        x_train1,y_train1,x_train2,y_train2,x_train3,y_train3,x_train4,y_train4=train.copy()
        data1,data2,data3,data4=data.copy()
        x1_train,y1_train=selectSample(x_train1,y_train1,labelsize)
        x2_train,y2_train=selectSample(x_train2,y_train2,labelsize)
        x3_train,y3_train=selectSample(x_train3,y_train3,labelsize)
        x4_train,y4_train=selectSample(x_train4,y_train4,labelsize)
        model1,model2,model3,model4=RandomForestClassifier(n_estimators=50),RandomForestClassifier(n_estimators=50),RandomForestClassifier(n_estimators=50),RandomForestClassifier(n_estimators=50)
        model1.fit(x1_train,y1_train.ravel())
        model2.fit(x2_train,y2_train.ravel())
        model3.fit(x3_train,y3_train.ravel())
        model4.fit(x4_train,y4_train.ravel())
        initial_result=report(test,model1,model2,model3,model4)
        x1_train_new,y1_train_new=selectSample(x_train1,y_train1,unlabelsize*epochs)
        x2_train_new,y2_train_new=selectSample(x_train2,y_train2,unlabelsize*epochs)
        x3_train_new,y3_train_new=selectSample(x_train3,y_train3,unlabelsize*epochs)
        x4_train_new,y4_train_new=selectSample(x_train4,y_train4,unlabelsize*epochs)
        x1_train=np.concatenate((x1_train,x1_train_new),axis=0)
        y1_train=np.concatenate((y1_train,y1_train_new),axis=0)
        x2_train=np.concatenate((x2_train,x2_train_new),axis=0)
        y2_train=np.concatenate((y2_train,y2_train_new),axis=0)
        x3_train=np.concatenate((x3_train,x3_train_new),axis=0)
        y3_train=np.concatenate((y3_train,y3_train_new),axis=0)
        x4_train=np.concatenate((x4_train,x4_train_new),axis=0)
        y4_train=np.concatenate((y4_train,y4_train_new),axis=0)
        model1.fit(x1_train,y1_train.ravel())
        model2.fit(x2_train,y2_train.ravel())
        model3.fit(x3_train,y3_train.ravel())
        model4.fit(x4_train,y4_train.ravel())
        multi_models=[model1,model2,model3,model4]
        final_result=report(test,model1,model2,model3,model4)
        resdf.iloc[sampletime*2,:24]=initial_result.ravel()
        resdf.iloc[sampletime*2+1,:24]=final_result.ravel()
        multi_ypreds,self_ypreds=[],[]
        for i in range(4):
            multi_ypreds.append(multi_models[i].predict(datareshapes[i]).astype(np.uint8).reshape(653,772))
            cv2.imwrite(os.path.join(map_dir,"multi_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_100class"+str(i)+"_"+str(sampletime)+".tif"),multi_ypreds[i])
    return resdf
#%%
map_dir=r""
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
datareshapes=[data1reshape,data2reshape,data3reshape,data4reshape]
# %%
# load label
label_path=r"G:\project\images\sentinal\process\labels"
label1_name,label2_name,label3_name,label4_name="train170211.csv","train170402.csv","train171009.csv","train171218.csv"
label1=np.loadtxt(open(os.path.join(label_path,label1_name),"r"),delimiter=',',skiprows=25,dtype=np.int)
label2=np.loadtxt(open(os.path.join(label_path,label2_name),"r"),delimiter=',',skiprows=25,dtype=np.int)
label3=np.loadtxt(open(os.path.join(label_path,label3_name),"r"),delimiter=',',skiprows=25,dtype=np.int)
label4=np.loadtxt(open(os.path.join(label_path,label4_name),"r"),delimiter=',',skiprows=25,dtype=np.int)
loc1=label1[:,:2]
loc2=label2[:,:2]
loc3=label3[:,:2]
loc4=label4[:,:2]
train1=addY(loc1,data1total,[1000]*n)
train2=addY(loc2,data2total,[1000]*n)
train3=addY(loc3,data3total,[1000]*n)
train4=addY(loc4,data4total,[1000]*n)
x_train1,x_test1,y_train1,y_test1=random_train_test_split(train1,size=1000)
x_train2,x_test2,y_train2,y_test2=random_train_test_split(train2,size=1000)
x_train3,x_test3,y_train3,y_test3=random_train_test_split(train3,size=1000)
x_train4,x_test4,y_train4,y_test4=random_train_test_split(train4,size=1000)
# %%
train=[x_train1,y_train1,x_train2,y_train2,x_train3,y_train3,x_train4,y_train4]
test=[x_test1,y_test1,x_test2,y_test2,x_train3,y_train3,x_train4,y_train4]
data=[data1reshape,data2reshape,data3reshape,data4reshape]
# %%
for labelsize in range(1,2,2):
    resdf=superrun(train,test,data,labelsize=labelsize*5,unlabelsize=10,epochs=10,sampletimes=50)
    #resdf.to_csv(r"G:\project\images\sentinal\process\output\res_csv\super\super_label{}_add100.csv".format(labelsize*5),index=False)
# %%
"""t-SNE进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
#%%

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1],
                 color=plt.cm.Set1(label[i] / 5.))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
#%%
data, label, n_samples, n_features = x_train1, y_train1.ravel(), x_train1.shape[0],x_train1.shape[1]
print('Computing t-SNE embedding')
tsne = TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
result = tsne.fit_transform(data)
fig = plot_embedding(result, label,
                        't-SNE embedding of the digits (time %.2fs)'
                        % (time() - t0))
plt.show(fig)
# %%
