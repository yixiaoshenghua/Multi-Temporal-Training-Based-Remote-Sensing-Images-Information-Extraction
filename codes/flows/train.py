#%%
import os

import cv2
import gdal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, log_loss)
from sklearn.model_selection import train_test_split


#%%
# function
## data process
def getBands(filename):# return (h,w,k)
    img = gdal.Open(filename)
    bands = []
    for i in range(img.RasterCount):
        band = img.GetRasterBand(i+1).ReadAsArray().astype(np.float)
        bands.append(band)
    bands = np.dstack((bands[0],bands[1],bands[2],bands[3]))
    return bands

def maxminNorm(data):# return (0,1)
    '''
    01标准化
    '''
    data=data.astype(np.float32)
    dmax=data.max()
    dmin=data.min()
    return (data-data.min())/(dmax-dmin)

def computeIndex(data,ind="NDVI"):
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
    
def returnIndex(data,indexName,df=False,reshape=True,norm=False):
    h,w,d=data.shape
    dataIdx=np.zeros((data.shape[0],data.shape[1],len(indexName)),dtype=np.float32)
    for i,idx in enumerate(indexName):
        if norm==True:
            dataIdx[:,:,i]=maxminNorm(computeIndex(data,ind=idx))
        else:
            dataIdx[:,:,i]=computeIndex(data,ind=idx)
    if len(np.unique(np.argwhere(np.isnan(dataIdx))[:,2]))!=0:
        print(np.unique(np.argwhere(np.isnan(dataIdx))[:,2]))
    if df==True:
        res=pd.DataFrame(dataIdx.reshape((h*w,-1)),columns=indexName)
    if reshape==True:
        res=dataIdx.reshape((h*w,-1))
    else:
        res=dataIdx
    return res

def returnData(img1,img2,indexName,df=False,reshape=False,norm=True):
    n=img1.shape[2]+len(indexName)
    data1=returnIndex(img1,indexName,reshape=False,norm=True)
    data2=returnIndex(img2,indexName,reshape=False,norm=True)
    data1=np.concatenate((maxminNorm(img1),data1),axis=2).reshape((-1,n))
    data2=np.concatenate((maxminNorm(img2),data2),axis=2).reshape((-1,n))
    return data1,data2

def returnUnlabeledData(data1,data2):
    diff_map=data1-data2
    diff_std=np.std(diff_map,axis=0)
    csd=np.sum(diff_map/diff_std,axis=1)
    unchanged_pos=np.where(csd<csd.mean())[0]
    unlabeled_data1,unlabeled_data2=data1[unchanged_pos,:],data2[unchanged_pos,:]
    return unlabeled_data1,unlabeled_data1

def addY(path,data,nums):#data2维
    loc=np.loadtxt(open(path,'r'),skiprows=25,delimiter=',',dtype=np.int)[:,:2]
    y=[]
    for i in range(len(nums)):
        y+=[i]*nums[i]
    y=np.array(y,dtype=np.int).reshape((-1,1))
    x=np.zeros((sum(nums),data.shape[1]),dtype=np.float32)
    for i,pos in enumerate(loc):
        x[i,:]=data[loc[i,1]*w+loc[i,0],:]
    labeled_data=np.concatenate((x,y),axis=1)
    return labeled_data

def selectSample(x,y,size=1,classSize=200):
    m=np.zeros((classSize*5),dtype=np.int)
    for i in range(5):
        m[i*classSize:(i+1)*classSize]=np.random.randint(i*classSize,(i+1)*classSize,size=size)
    return x[m,:],y[m]

## train
def resultReport(model1,model2,model3,model4,X_test1,y_test1,X_test2,y_test2,unlabeled_data1,unlabeled_data2):
    y_pred1=model1.predict(X_test1)
    y_pred2=model2.predict(X_test2)
    y_pred3=model3.predict(X_test1)
    y_pred4=model4.predict(X_test2)
    res1=classification_report(y_test1,y_pred1,
                                target_names=['veg','shadow','water','road','building'],
                                output_dict=True)
    res2=classification_report(y_test2,y_pred2,
                                target_names=['veg','shadow','water','road','building'],
                                output_dict=True)
    res3=classification_report(y_test1,y_pred3,
                                target_names=['veg','shadow','water','road','building'],
                                output_dict=True)
    res4=classification_report(y_test1,y_pred4,
                                target_names=['veg','shadow','water','road','building'],
                                output_dict=True)
    res=np.array([res1['veg']['f1-score'],res1['shadow']['f1-score'],res1['water']['f1-score'],
                  res1['road']['f1-score'],res1['building']['f1-score'],res1['weighted avg']['f1-score'],
                  res2['veg']['f1-score'],res2['shadow']['f1-score'],res2['water']['f1-score'],
                  res2['road']['f1-score'],res2['building']['f1-score'],res2['weighted avg']['f1-score'],
                  res3['veg']['f1-score'],res3['shadow']['f1-score'],res3['water']['f1-score'],
                  res3['road']['f1-score'],res3['building']['f1-score'],res3['weighted avg']['f1-score'],
                  res4['veg']['f1-score'],res4['shadow']['f1-score'],res4['water']['f1-score'],
                  res4['road']['f1-score'],res4['building']['f1-score'],res4['weighted avg']['f1-score'],
                  calcPDC(model1,model2,unlabeled_data1,unlabeled_data2),calcPDC(model3,model4,unlabeled_data1,unlabeled_data2)
                  ])
    return res

def calcPDC(model1,model2,unlabeled_data1,unlabeled_data2):
    Nu=unlabeled_data1.shape[0]
    y_pred1_unlabeled=model1.predict(unlabeled_data1)
    y_pred2_unlabeled=model2.predict(unlabeled_data2)
    pdc=np.sum(y_pred1_unlabeled-y_pred2_unlabeled!=0)
    return pdc/Nu

def co_train(X1,y1,X2,y2,unlabeled_data1,unlabeled_data2,model1,model2,size,c):
    n=np.unique(y1).shape[0]
    features1=['blue1','grn1','red1','nir1','NDVI1','NDWI1','MSAVI1','MTVI1','VARI1']
    features2=['blue2','grn2','red2','nir2','NDVI2','NDWI2','MSAVI2','MTVI2','VARI2']
    datadf=pd.DataFrame(np.zeros((unlabeled_data1.shape[0],unlabeled_data1.shape[1]+unlabeled_data2.shape[1]+5)),
            columns=['blue1','grn1','red1','nir1','NDVI1','NDWI1','MSAVI1','MTVI1','VARI1',
                     'blue2','grn2','red2','nir2','NDVI2','NDWI2','MSAVI2','MTVI2','VARI2',
                     'y_pred1','y_pred2','y_proba1','y_proba2','selected'])
    datadf.loc[:,:9],datadf.loc[:,9:18]=unlabeled_data1,unlabeled_data2
    for epoch in range(epochs):
        datadf['y_pred1']=model1.predict(unlabeled_data1)
        datadf['y_pred2']=model2.predict(unlabeled_data2)
        datadf['y_proba1']=model1.predict_proba(unlabeled_data1).max(axis=1)
        datadf['y_proba2']=model2.predict_proba(unlabeled_data2).max(axis=1)
        sizes=[size]*n
        # 找到guessclass一致区域的数据
        datadf_candidate=datadf[datadf['y_pred1']==datadf['y_pred2']]
        # 标签一致区域找到置信度最高的区域
        class_candidate=[datadf_candidate[datadf_candidate['y_pred1']==i] for i in range(n)]
        # 找到两期置信度都高的区域更新训练集
        for i in range(n):
            # tradeoff
            th1=c*class_candidate[i]['y_proba1'].mean()
            th2=c*class_candidate[i]['y_proba2'].mean()
            print(th1,th2)
            index1=np.array(class_candidate[i][class_candidate[i]['y_proba1']>th1].index)
            index2=np.array(class_candidate[i][class_candidate[i]['y_proba2']>th2].index)
            index=np.array([x for x in index1 if x in index2])
            print(index1,index2,index)
            sample1=np.random.randint(low=0,high=int(index.shape[0]),size=int(sizes[i]))
            sample2=np.random.randint(low=0,high=int(index.shape[0]),size=int(sizes[i]))
            can1=index1[sample1]
            can2=index2[sample2]
            X1=np.concatenate((X1,class_candidate[i].loc[can1,features1]),axis=0)
            X2=np.concatenate((X2,class_candidate[i].loc[can2,features2]),axis=0)
            y1=np.concatenate((y1,class_candidate[i].loc[can1,['y_pred1']].values.ravel()),axis=0)
            y2=np.concatenate((y2,class_candidate[i].loc[can2,['y_pred2']].values.ravel()),axis=0)
            model1.fit(X1,y1)
            model2.fit(X2,y2)
    return model1,model2

def self_train(X1,y1,X2,y2,unlabeled_data1,unlabeled_data2,model1,model2,size,c):
    n=np.unique(y1).shape[0]
    features1=['blue1','grn1','red1','nir1','NDVI1','NDWI1','MSAVI1','MTVI1','VARI1']
    features2=['blue2','grn2','red2','nir2','NDVI2','NDWI2','MSAVI2','MTVI2','VARI2']
    datadf=pd.DataFrame(np.zeros((unlabeled_data1.shape[0],unlabeled_data1.shape[1]+unlabeled_data2.shape[1]+5)),
            columns=['blue1','grn1','red1','nir1','NDVI1','NDWI1','MSAVI1','MTVI1','VARI1',
                     'blue2','grn2','red2','nir2','NDVI2','NDWI2','MSAVI2','MTVI2','VARI2',
                     'y_pred1','y_pred2','y_proba1','y_proba2','selected'])
    datadf.loc[:,:9],datadf.loc[:,9:18]=unlabeled_data1,unlabeled_data2
    for epoch in range(epochs):
        datadf['y_pred1']=model1.predict(unlabeled_data1)
        datadf['y_pred2']=model2.predict(unlabeled_data2)
        datadf['y_proba1']=model1.predict_proba(unlabeled_data1).max(axis=1)
        datadf['y_proba2']=model2.predict_proba(unlabeled_data2).max(axis=1)
        sizes=[size]*n
        class_candidate1=[datadf[datadf['y_pred1']==i] for i in range(n)]
        class_candidate2=[datadf[datadf['y_pred2']==i] for i in range(n)]
        # 找到两期置信度都高的区域更新训练集
        for i in range(n):
            # tradeoff
            th1=c*class_candidate1[i]['y_proba1'].mean()
            th2=c*class_candidate2[i]['y_proba2'].mean()
            index1=np.array(class_candidate1[i][class_candidate1[i]['y_proba1']>th1].index)
            index2=np.array(class_candidate2[i][class_candidate2[i]['y_proba2']>th2].index)
            index=np.array([x for x in index1 if x in index2])
            sample1=np.random.randint(low=0,high=int(index1.shape[0]),size=int(sizes[i]))
            sample2=np.random.randint(low=0,high=int(index2.shape[0]),size=int(sizes[i]))
            can1=index1[sample1]
            can2=index2[sample2]
            X1=np.concatenate((X1,class_candidate1[i].loc[can1,features1]),axis=0)
            X2=np.concatenate((X2,class_candidate2[i].loc[can2,features2]),axis=0)
            y1=np.concatenate((y1,class_candidate1[i].loc[can1,['y_pred1']].values.ravel()),axis=0)
            y2=np.concatenate((y2,class_candidate2[i].loc[can2,['y_pred2']].values.ravel()),axis=0)
            model1.fit(X1,y1)
            model2.fit(X2,y2)
    return model1,model2

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
    return X_train, X_test, y_train, y_test



#%%
# data prepare
## get two temporal image data
image_dir=r'G:\project\images\sentinal'
img1=getBands(os.path.join(image_dir,'161103.tif'))
img2=getBands(os.path.join(image_dir,'170211.tif'))
h,w,_=img1.shape
## compute and return two temporal features
indexName=['NDVI','NDWI','MSAVI','MTVI','VARI']
data1,data2=returnData(img1,img2,indexName) # (h*w,9)
## calculate the std of each bands, find the unchanged area
unlabeled_data1,unlabeled_data2=returnUnlabeledData(data1,data2) # (n,9)
## prepare train_test data
path1,path2=r'G:\project\images\sentinal\process\labels\train161103.csv',r'G:\project\images\sentinal\process\labels\train170211.csv'
labeled_data1,labeled_data2=addY(path1,data1,[1000]*5),addY(path2,data2,[1000]*5)

#%%
# train
epochs=10
sample_times=50
labeled_data_epoch_size_max=10
labeled_data_epoch_size_min=1
unlabeled_data_epoch_size_max=55
unlabeled_data_epoch_size_min=5
resdf=pd.DataFrame(np.zeros((sample_times*2*10*5,26)),
                    columns=['co_veg1','co_shadow1','co_water1','co_road1','co_building1','co_ave1',
                             'co_veg2','co_shadow2','co_water2','co_road2','co_building2','co_ave2',
                             'se_veg1','se_shadow1','se_water1','se_road1','se_building1','se_ave1',
                             'se_veg2','se_shadow2','se_water2','se_road2','se_building2','se_ave2',
                             'co_pdc','se_pdc'])
i=0
for labeled_data_epoch_size in range(labeled_data_epoch_size_min,labeled_data_epoch_size_max,2):
    j=0
    for unlabeled_data_epoch_size in range(unlabeled_data_epoch_size_min,unlabeled_data_epoch_size_max,5):
        X_train1, X_test1, y_train1, y_test1 = random_train_test_split(labeled_data1,size=1000,train_rate=0.2,test_rate=0.8)
        X_train2, X_test2, y_train2, y_test2 = random_train_test_split(labeled_data2,size=1000,train_rate=0.2,test_rate=0.8)
        k=0
        for sample_id in range(sample_times):
            print("i:%d,j:%d,k:%d"%(i,j,k))
            X1,y1=selectSample(X_train1,y_train1,size=labeled_data_epoch_size,classSize=200)
            X2,y2=selectSample(X_train2,y_train2,size=labeled_data_epoch_size,classSize=200)
            model1=RandomForestClassifier(n_estimators=50)
            model2=RandomForestClassifier(n_estimators=50)
            model3=RandomForestClassifier(n_estimators=50)
            model4=RandomForestClassifier(n_estimators=50)
            model1.fit(X1,y1)
            model2.fit(X2,y2)
            model3.fit(X1,y1)
            model4.fit(X2,y2)

            pre_res=resultReport(model1,model2,model3,model4,X_test1,y_test1,X_test2,y_test2,unlabeled_data1.copy(),unlabeled_data2.copy())
            resdf.loc[i*5+j**10+k*2,:]=pre_res
            # co-train and self-train
            model1,model2=co_train(X1.copy(),y1.copy(),X2.copy(),y2.copy(),unlabeled_data1.copy(),unlabeled_data2.copy(),model1,model2,size=unlabeled_data_epoch_size,c=1.0)
            model3,model4=self_train(X1.copy(),y1.copy(),X2.copy(),y2.copy(),unlabeled_data1.copy(),unlabeled_data2.copy(),model3,model4,size=unlabeled_data_epoch_size,c=1.0)
            post_res=resultReport(model1,model2,model3,model4,X_test1,y_test1,X_test2,y_test2,unlabeled_data1.copy(),unlabeled_data2.copy())
            resdf.loc[i*5+j*10+k*2+1,:]=post_res
            print('initial:',pre_res[5],pre_res[11],pre_res[17],pre_res[23])
            print('final:',post_res[5],post_res[11],post_res[17],post_res[23])
            # predict whole map
            map1,map2,map3,map4=model1.predict(data1),model2.predict(data2),model3.predict(data1),model4.predict(data2)
            cv2.imwrite(r'G:\project\images\sentinal\process\output\map\161103_170211\co_label_'+str(labeled_data_epoch_size)+'unlabel_'+str(unlabeled_data_epoch_size)+'sampleid_'+str(sample_id)+'year1.tif',map1.astype(np.uint8).reshape((h,w)))
            cv2.imwrite(r'G:\project\images\sentinal\process\output\map\161103_170211\co_label_'+str(labeled_data_epoch_size)+'unlabel_'+str(unlabeled_data_epoch_size)+'sampleid_'+str(sample_id)+'year2.tif',map2.astype(np.uint8).reshape((h,w)))
            cv2.imwrite(r'G:\project\images\sentinal\process\output\map\161103_170211\se_label_'+str(labeled_data_epoch_size)+'unlabel_'+str(unlabeled_data_epoch_size)+'sampleid_'+str(sample_id)+'year1.tif',map3.astype(np.uint8).reshape((h,w)))
            cv2.imwrite(r'G:\project\images\sentinal\process\output\map\161103_170211\se_label_'+str(labeled_data_epoch_size)+'unlabel_'+str(unlabeled_data_epoch_size)+'sampleid_'+str(sample_id)+'year2.tif',map4.astype(np.uint8).reshape((h,w)))
            if k==0:
                break
            k+=1
        if j==0:
            break
        j+=1
    if i==0:
        break
    i+=1
# post process
resdf.to_csv(r"G:\project\images\sentinal\process\output\res\161103_170211_epochs10_samples50_labeleddata13579_unlabeleddata5_c1.csv")


# %%
