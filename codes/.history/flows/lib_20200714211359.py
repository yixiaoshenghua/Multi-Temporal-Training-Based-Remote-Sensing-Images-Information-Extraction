#%%
import gdal
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,f1_score,classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier

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
    y=np.array([0]*nums[0]+[1]*nums[1]+[2]*nums[2]).reshape((sum(nums),1))
    x=np.zeros((sum(nums),data.shape[2]))
    for i,yx in enumerate(loc):
        x[i,:]=data[yx[1],yx[0],:]
    train=np.concatenate((x,y),axis=1)
    return train

# def cotrain(datadf,x1,y1,x2,y2,size=10,selflearn=False):
#     centers=np.zeros((8,x1.shape[1]),dtype=np.float32)
#     k=x1.shape[0]//4
#     for i in range(4):
#         centers[i,:]=np.mean(x1[i*k:(i+1)*k,:])
#         centers[4+i,:]=np.mean(x2[i*k:(i+1)*k,:])
#     lst=['dis11','dis12','dis13','dis14','dis21','dis22','dis23','dis24','dis1Mesure','dis2Mesure','dis3Mesure','dis4Mesure']
#     for i in range(4):
#         datadf[lst[i]]=distance(datadf.loc[:,features1],centers[i])
#         datadf[lst[4+i]]=distance(datadf.loc[:,features2],centers[4+i])
#         datadf[lst[8+i]]=datadf[lst[i]]**2+datadf[lst[4+i]]**2
#     datadf_candidate=datadf[datadf['change1']==0]
#     sizes=[size]*4
#     if selflearn==False:
#         for i in range(4):
#             index=np.array(datadf_candidate.sort_values(by=lst[8+i]).index)
#             can=index[:sizes[i]]
#             x1=np.concatenate((x1,datadf_candidate.loc[can,features1]),axis=0)
#             x2=np.concatenate((x2,datadf_candidate.loc[can,features2]),axis=0)
#             y1=np.concatenate((y1,np.zeros((sizes[i],1))+i),axis=0)
#             y2=np.concatenate((y2,np.zeros((sizes[i],1))+i),axis=0)
#     else:
#         for i in range(4):
#             index1=np.array(datadf_candidate.sort_values(by=lst[i]).index)
#             index2=np.array(datadf_candidate.sort_values(by=lst[4+i]).index)
#             can1=index1[:sizes[i]]
#             can2=index2[:sizes[i]]
#             x1=np.concatenate((x1,datadf_candidate.loc[can1,features1]),axis=0)
#             x2=np.concatenate((x2,datadf_candidate.loc[can2,features2]),axis=0)
#             y1=np.concatenate((y1,np.zeros((sizes[i],1))+i),axis=0)
#             y2=np.concatenate((y2,np.zeros((sizes[i],1))+i),axis=0)
#     return x1,y1,x2,y2
# def cotrain(datadf,x1,y1,x2,y2,model1,model2,size=50,selflearn=False):
#     # 候选预测数据概率和标签
#     datadf['prob1']=model1.predict_proba(datadf[features1]).max(axis=1)
#     datadf['prob2']=model2.predict_proba(datadf[features2]).max(axis=1)
#     datadf['guessClass1']=model1.predict(datadf[features1])
#     datadf['guessClass2']=model2.predict(datadf[features2])
#     sizes=[size]*4
#     if selflearn==False:
#         # 找到guessclass一致区域的数据
#         datadf_candidate=datadf[datadf['change1']==0][datadf['guessClass1']==datadf['guessClass2']]
#         # 标签一致区域找到置信度最高的区域
#         class_candidate=[datadf_candidate[datadf_candidate['guessClass1']==i] for i in range(5)]
#         # 找到两期置信度都高的区域更新训练集
#         for i in range(4):
#             index1=np.array(class_candidate[i].sort_values(by="prob1" , ascending=False).index)
#             index2=np.array(class_candidate[i].sort_values(by="prob2" , ascending=False).index)
#             can1=index1[:sizes[i]]
#             can2=index2[:sizes[i]]
#             x1=np.concatenate((x1,class_candidate[i].loc[can1,features1]),axis=0)
#             x2=np.concatenate((x2,class_candidate[i].loc[can2,features2]),axis=0)
#             y1=np.concatenate((y1,class_candidate[i].loc[can1,['guessClass1']].values.reshape((-1,1))),axis=0)
#             y2=np.concatenate((y2,class_candidate[i].loc[can2,['guessClass2']].values.reshape((-1,1))),axis=0)
#     else:
#         class_candidate1=[datadf[datadf['guessClass1']==i] for i in range(5)]
#         class_candidate2=[datadf[datadf['guessClass2']==i] for i in range(5)]
#         for i in range(4):
#             index1=np.array(class_candidate1[i].sort_values(by="prob1" , ascending=False).index)
#             index2=np.array(class_candidate2[i].sort_values(by="prob2" , ascending=False).index)
#             can1=index1[:sizes[i]]
#             can2=index2[:sizes[i]]
#             x1=np.concatenate((x1,class_candidate1[i].loc[can1,features1]),axis=0)
#             x2=np.concatenate((x2,class_candidate2[i].loc[can2,features2]),axis=0)
#             y1=np.concatenate((y1,class_candidate1[i].loc[can1,['guessClass1']].values.reshape((-1,1))),axis=0)
#             y2=np.concatenate((y2,class_candidate2[i].loc[can2,['guessClass2']].values.reshape((-1,1))),axis=0)
#     return x1,y1,x2,y2,model1,model2
def cotrain(datadf,x1,y1,x2,y2,model1,model2,c=1.0,size=50,selflearn=False):
    # 候选预测数据概率和标签
    datadf['prob1']=model1.predict_proba(datadf[features1]).max(axis=1)
    datadf['prob2']=model2.predict_proba(datadf[features2]).max(axis=1)
    datadf['guessClass1']=model1.predict(datadf[features1])
    datadf['guessClass2']=model2.predict(datadf[features2])
    sizes=[size]*3
    if selflearn==False:
        # 找到guessclass一致区域的数据
        datadf_candidate=datadf[datadf['change1']==0][datadf['guessClass1']==datadf['guessClass2']]
        # 标签一致区域找到置信度最高的区域
        class_candidate=[datadf_candidate[datadf_candidate['guessClass1']==i] for i in range(3)]
        # 找到两期置信度都高的区域更新训练集
        for i in range(3):
            # tradeoff
            th1=class_candidate[i]['prob1'].mean()*c
            th2=class_candidate[i]['prob2'].mean()*c
            index1=np.array(class_candidate[i][class_candidate[i]['prob1']>th1].index)
            index2=np.array(class_candidate[i][class_candidate[i]['prob2']>th2].index)
            index=np.array([x for x in index1 if x in index2])
            sample1=np.random.randint(int(index.shape[0]),size=int(sizes[i]))
            sample2=np.random.randint(int(index.shape[0]),size=int(sizes[i]))
            can1=index1[sample1]
            can2=index2[sample2]
            x1=np.concatenate((x1,class_candidate[i].loc[can1,features1]),axis=0)
            x2=np.concatenate((x2,class_candidate[i].loc[can2,features2]),axis=0)
            y1=np.concatenate((y1,class_candidate[i].loc[can1,['guessClass1']].values.reshape((-1,1))),axis=0)
            y2=np.concatenate((y2,class_candidate[i].loc[can2,['guessClass2']].values.reshape((-1,1))),axis=0)
    else:
        class_candidate1=[datadf[datadf['guessClass1']==i] for i in range(3)]
        class_candidate2=[datadf[datadf['guessClass2']==i] for i in range(3)]
        for i in range(3):
            th1=class_candidate[i]['prob1'].mean()*c
            th2=class_candidate[i]['prob2'].mean()*c
            index1=np.array(class_candidate[i][class_candidate[i]['prob1']>th1].index)
            index2=np.array(class_candidate[i][class_candidate[i]['prob2']>th2].index)
            sample1=np.random.randint(int(index1.shape[0]),size=int(sizes[i]))
            sample2=np.random.randint(int(index2.shape[0]),size=int(sizes[i]))
            can1=index1[sample1]
            can2=index2[sample2]
            x1=np.concatenate((x1,class_candidate1[i].loc[can1,features1]),axis=0)
            x2=np.concatenate((x2,class_candidate2[i].loc[can2,features2]),axis=0)
            y1=np.concatenate((y1,class_candidate1[i].loc[can1,['guessClass1']].values.reshape((-1,1))),axis=0)
            y2=np.concatenate((y2,class_candidate2[i].loc[can2,['guessClass2']].values.reshape((-1,1))),axis=0)
    return x1,y1,x2,y2,model1,model2
def run(datadf,x1,y1,x2,y2,model1,model2,x_test1,y_test1,x_test2,y_test2,size=50,epoch=20,selflearn=False):
    if selflearn==False:
        crossScore1,crossScore2=[],[]
        for i in range(epoch):
            model1.fit(x1,y1.ravel())
            model2.fit(x2,y2.ravel())
            y_pre1 = model1.predict(x_test1)
            y_pre2 = model2.predict(x_test2)
            x1,y1,x2,y2,model1,model2=cotrain(datadf,x1,y1,x2,y2,model1,model2,size,selflearn)
            res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
            res2=classification_report(y_test2,y_pre2,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
            f1=res1['veg']['f1-score']
            f2=res2['veg']['f1-score']
            # f1=f1_score(y_test1, y_pre1, average='weighted')
            # f2=f1_score(y_test2, y_pre2, average='weighted')
            if i%10==0:
                print("score: f1:%.5f f2:%.5f"%(f1,f2))
            crossScore1.append(f1)
            crossScore2.append(f2)
        return model1,model2,crossScore1,crossScore2
    else:
        selfScore1,selfScore2=[],[]
        for i in range(epoch):
            model1.fit(x1,y1.ravel())
            model2.fit(x2,y2.ravel())
            y_pre1 = model1.predict(x_test1)
            y_pre2 = model2.predict(x_test2)
            x1,y1,x2,y2,model1,model2=cotrain(datadf,x1,y1,x2,y2,model1,model2,size,selflearn)
            res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
            res2=classification_report(y_test2,y_pre2,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
            f1=res1['veg']['f1-score']
            f2=res2['veg']['f1-score']
            # f1=f1_score(y_test1, y_pre1, average='weighted')
            # f2=f1_score(y_test2, y_pre2, average='weighted')
            if i%10==0:
                print("score: f1:%.5f f2:%.5f"%(f1,f2))
            selfScore1.append(f1)
            selfScore2.append(f2)
        return model1,model2,selfScore1,selfScore2

def selectSample(x,y,sample):
    m=np.zeros((sample*3),dtype=np.int)
    for i in range(3):
        m[i*sample:(i+1)*sample]=np.random.randint(i*50,(i+1)*50,size=sample)
    return x[m,:],y[m,:]    

def color(y_pred):
    y_pred=y_pred.reshape((653,772))
    show=np.zeros((653,772,3))
    cmap=np.array([[0,255,0],[255,255,0],[255,0,0],[0,255,255]])
    for i in range(3):
        show[y_pred==i,:]=cmap[i]
    return show

def distance(data,center):
    dis=np.sqrt(np.sum((data-center)**2))
    return dis


# %%
