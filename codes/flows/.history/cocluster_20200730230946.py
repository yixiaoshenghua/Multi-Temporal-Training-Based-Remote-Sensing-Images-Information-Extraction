#%%
#from lib import *
#%%
import gdal
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.metrics import log_loss,confusion_matrix,plot_confusion_matrix,f1_score,classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
import scipy.stats
from trees.dt import *
from trees.rf import *
from trees.gbdt import *
from trees_ori import gbdt
#%%

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

def cotrain(datadf,x1,y1,x2,y2,model1,model2,c=1.0,size=5,selflearn=False):
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
            th1=c*class_candidate[i]['prob1'].mean()
            th2=c*class_candidate[i]['prob2'].mean()
            index1=np.array(class_candidate[i][class_candidate[i]['prob1']>th1].index)
            index2=np.array(class_candidate[i][class_candidate[i]['prob2']>th2].index)
            index=np.array([x for x in index1 if x in index2])
            sample1=np.random.randint(low=0,high=int(index.shape[0]),size=int(sizes[i]))
            sample2=np.random.randint(low=0,high=int(index.shape[0]),size=int(sizes[i]))
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
            th1=c*class_candidate1[i]['prob1'].mean()
            th2=c*class_candidate2[i]['prob2'].mean()
            index1=np.array(class_candidate1[i][class_candidate1[i]['prob1']>th1].index)
            index2=np.array(class_candidate2[i][class_candidate2[i]['prob2']>th2].index)
            sample1=np.random.randint(low=0,high=int(index1.shape[0]),size=int(sizes[i]))
            sample2=np.random.randint(low=0,high=int(index2.shape[0]),size=int(sizes[i]))
            can1=index1[sample1]
            can2=index2[sample2]
            x1=np.concatenate((x1,class_candidate1[i].loc[can1,features1]),axis=0)
            x2=np.concatenate((x2,class_candidate2[i].loc[can2,features2]),axis=0)
            y1=np.concatenate((y1,class_candidate1[i].loc[can1,['guessClass1']].values.reshape((-1,1))),axis=0)
            y2=np.concatenate((y2,class_candidate2[i].loc[can2,['guessClass2']].values.reshape((-1,1))),axis=0)
    return x1,y1,x2,y2,model1,model2

def run(resdf,datadf,x1,y1,x2,y2,model1,model2,x_test1,y_test1,x_test2,y_test2,c=1.0,size=5,epoch=20,selflearn=False,curepoch=0):
    unlabeled_X_train1=datadf[datadf['change1']==0].loc[:,features1].values
    unlabeled_X_train2=datadf[datadf['change2']==0].loc[:,features2].values
    if selflearn==False:
        for i in range(epoch):
            ii=curepoch*epoch+i
            unlabeled_X_batch_train1,unlabeled_X_batch_train2=selectSample(unlabeled_X_train1,unlabeled_X_train2,sample=size)
            model1.fit(x1,y1.ravel().astype('int64'),x2,y2.ravel().astype('int64'),unlabeled_X_batch_train1,unlabeled_X_batch_train2)
            model2.fit(x2,y2.ravel().astype('int64'),x1,y1.ravel().astype('int64'),unlabeled_X_batch_train2,unlabeled_X_batch_train1)
            y_pre1 = model1.predict(x_test1)
            y_pre2 = model2.predict(x_test2)
            pred1=model1.predict(datadf[datadf['change1']==0].loc[:,features1].values)
            pred2=model2.predict(datadf[datadf['change1']==0].loc[:,features2].values)
            y_pred1=model1.predict_proba(datadf[datadf['change1']==0].loc[:,features1].values)
            y_pred2=model2.predict_proba(datadf[datadf['change1']==0].loc[:,features2].values)
            y1_pred=model1.predict(datadf.loc[:,features1].values)
            y2_pred=model2.predict(datadf.loc[:,features2].values)
            resdf.at[ii,'coLoss1']=calcConsistencyLoss(datadf,y_pred1,pred2)
            resdf.at[ii,'coLoss2']=calcConsistencyLoss(datadf,y_pred2,pred1)
            resdf.at[ii,'coPDC']=calcPDC(datadf,y1_pred,y2_pred)
            x1,y1,x2,y2,model1,model2=cotrain(datadf,x1,y1,x2,y2,model1,model2,c,size,selflearn)
            res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
            res2=classification_report(y_test2,y_pre2,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
            f1=f1_score(y_test1, y_pre1, average='weighted')
            f2=f1_score(y_test2, y_pre2, average='weighted')
            resdf.iat[ii,0],resdf.iat[ii,1],resdf.iat[ii,2],resdf.iat[ii,3]=res1['veg']['f1-score'],res1['water']['f1-score'],res1['building']['f1-score'],f1
            resdf.iat[ii,4],resdf.iat[ii,5],resdf.iat[ii,6],resdf.iat[ii,7]=res2['veg']['f1-score'],res2['water']['f1-score'],res2['building']['f1-score'],f2
            if i%5==0:
                print("score: f1:%.5f f2:%.5f"%(f1,f2))
        return model1,model2,resdf
    else:
        for i in range(epoch):
            ii=curepoch*epoch+i
            unlabeled_X_batch_train1,unlabeled_X_batch_train2=selectSample(unlabeled_X_train1,unlabeled_X_train2,sample=size)
            model1.fit(x1,y1.ravel().astype('int64'),x2,y2.ravel().astype('int64'),unlabeled_X_batch_train1,unlabeled_X_batch_train2)
            model2.fit(x2,y2.ravel().astype('int64'),x1,y1.ravel().astype('int64'),unlabeled_X_batch_train2,unlabeled_X_batch_train1)
            y_pre1 = model1.predict(x_test1)
            y_pre2 = model2.predict(x_test2)
            pred1=model1.predict(datadf[datadf['change1']==0].loc[:,features1].values)
            pred2=model2.predict(datadf[datadf['change1']==0].loc[:,features2].values)
            y_pred1=model1.predict_proba(datadf[datadf['change1']==0].loc[:,features1].values)
            y_pred2=model2.predict_proba(datadf[datadf['change1']==0].loc[:,features2].values)
            y1_pred=model1.predict(datadf.loc[:,features1].values)
            y2_pred=model2.predict(datadf.loc[:,features2].values)
            resdf.at[ii,'seLoss1']=calcConsistencyLoss(datadf,y_pred1,pred2)
            resdf.at[ii,'seLoss2']=calcConsistencyLoss(datadf,y_pred2,pred1)
            resdf.at[ii,'sePDC']=calcPDC(datadf,y1_pred,y2_pred)
            x1,y1,x2,y2,model1,model2=cotrain(datadf,x1,y1,x2,y2,model1,model2,c,size,selflearn)
            res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
            res2=classification_report(y_test2,y_pre2,
                                    target_names=['veg','water','building'],
                                    output_dict=True)
            f1=f1_score(y_test1, y_pre1, average='weighted')
            f2=f1_score(y_test2, y_pre2, average='weighted')
            resdf.iat[ii,8],resdf.iat[ii,9],resdf.iat[ii,10],resdf.iat[ii,11]=res1['veg']['f1-score'],res1['water']['f1-score'],res1['building']['f1-score'],f1
            resdf.iat[ii,12],resdf.iat[ii,13],resdf.iat[ii,14],resdf.iat[ii,15]=res2['veg']['f1-score'],res2['water']['f1-score'],res2['building']['f1-score'],f2
            if i%5==0:
                print("score: f1:%.5f f2:%.5f"%(f1,f2))

        return model1,model2,resdf

def runModel(datadf,x1,y1,x2,y2,x_test1,y_test1,x_test2,y_test2,c=1.0,size=5,epoch=20,labelsize=1,sampletimes=10):
    resdf=pd.DataFrame(np.zeros((sampletimes*epoch,22)),
                    columns=['17co'+str(labelsize)+'veg','17co'+str(labelsize)+'water','17co'+str(labelsize)+'building','17co'+str(labelsize)+'total',
                            '19co'+str(labelsize)+'veg','19co'+str(labelsize)+'water','19co'+str(labelsize)+'building','19co'+str(labelsize)+'total',
                            '17se'+str(labelsize)+'veg','17se'+str(labelsize)+'water','17se'+str(labelsize)+'building','17se'+str(labelsize)+'total',
                            '19se'+str(labelsize)+'veg','19se'+str(labelsize)+'water','19se'+str(labelsize)+'building','19se'+str(labelsize)+'total',
                            'coPDC','sePDC','coLoss1','coLoss2','seLoss1','seLoss2'])
    # 构造model训练
    for i in range(sampletimes):
        x1_train,y1_train=selectSample(x1,y1,labelsize)
        x2_train,y2_train=selectSample(x2,y2,labelsize)
        print("%d epoch"%i)
        # model1=RandomForestClassifier(n_estimators=50)
        # model2=RandomForestClassifier(n_estimators=50)
        # model1,model2=initModel(x1_train,y1_train,x2_train,y2_train,datadf)
        model1=GradientBoostedDecisionTree(n_iter=20,learning_rate=0.1)
        model2=GradientBoostedDecisionTree(n_iter=20,learning_rate=0.1)
        x_train1=x1_train.copy()
        x_train2=x2_train.copy()
        y_train1=y1_train.copy()
        y_train2=y2_train.copy()
        model1,model2,resdf=run(resdf,datadf,x_train1,y_train1,x_train2,y_train2,model1,model2,x_test1,y_test1,x_test2,y_test2,c=1.0,size=size,epoch=epoch,selflearn=False,curepoch=i)
        y_pred1=model1.predict(data1reshape).astype(np.int)
        show1=color(y_pred1)
        cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\cosample"+str(labelsize)+"epoch20size5class17_"+str(i)+".jpg",show1)
        y_pred2=model2.predict(data2reshape).astype(np.int)
        show2=color(y_pred2)
        cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\cosample"+str(labelsize)+"epoch20size5class19_"+str(i)+".jpg",show2)
        print("%d epoch"%i)
        # model3=RandomForestClassifier(n_estimators=50)
        # model4=RandomForestClassifier(n_estimators=50)
        # model3,model4=initModel(x1_train,y1_train,x2_train,y2_train,datadf)
        model3=GradientBoostedDecisionTree(n_iter=20,learning_rate=0.1)
        model4=GradientBoostedDecisionTree(n_iter=20,learning_rate=0.1)
        x_train3=x1_train.copy()
        x_train4=x2_train.copy()
        y_train3=y1_train.copy()
        y_train4=y2_train.copy()
        model3,model4,resdf=run(resdf,datadf,x_train3,y_train3,x_train4,y_train4,model3,model4,x_test1,y_test1,x_test2,y_test2,c=1.0,size=size,epoch=epoch,selflearn=True,curepoch=i)
        y_pred3=model3.predict(data1reshape).astype(np.int)
        show3=color(y_pred3)
        cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\selfsample"+str(labelsize)+"epoch20size5class17_"+str(i)+".jpg",show3)
        y_pred4=model2.predict(data2reshape).astype(np.int)
        show4=color(y_pred4)
        cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\selfsample"+str(labelsize)+"epoch20size5class19_"+str(i)+".jpg",show4)
    return resdf


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

def calcPDC(datadf,pred1,pred2):
    unchangedf=datadf[datadf['change1']==0]
    Nu=unchangedf.shape[0]
    pd=(pred1-pred2).reshape(653,772)
    pdc=0
    for i in range(unchangedf.shape[0]):
        if pd[unchangedf.iloc[i,10],unchangedf.iloc[i,11]]!=0:
            pdc+=1
    return pdc/Nu

def calcConsistencyLoss(datadf,y_pred1,y2_pred,lamda=1.0):
    # labeled_loss=log_loss(y_true,y_pred,labels=[0,1,2])
    unlabeled_loss=log_loss(y2_pred,y_pred1,labels=[0,1,2])
    # loss=labeled_loss+unlabeled_loss*lamda
    return unlabeled_loss

def initModel(x1_train,y1_train,x2_train,y2_train,datadf):
    model1=RandomForestClassifier(n_estimators=50)
    model2=RandomForestClassifier(n_estimators=50)
    model1.fit(x1_train,y1_train)
    model2.fit(x2_train,y2_train)
    unlabeled_x1=datadf[datadf['change1']==0].loc[:,features1].values
    unlabeled_x2=datadf[datadf['change2']==0].loc[:,features2].values
    unlabeled_y1=model1.predict(unlabeled_x1)
    unlabeled_y2=model2.predict(unlabeled_x2)
    model1.fit(unlabeled_x1,unlabeled_y2)
    model2.fit(unlabeled_x2,unlabeled_y1)
    return model1,model2

def consistencyLoss(y1,y2):
    """
    Compute the consistency loss of two temporal predictions
    """
    hist1=np.bincount(y1)
    hist2=np.bincount(y2)
    hist=np.vstack([hist1,hist2])
    cnt=np.min(hist,axis=0)
    loss=np.sum(cnt/hist[0,:]*cnt/hist[1,:])
    return loss

# def crossConsistencyLoss(yProb1,yProb2):
#     """
#     Compute the consistency crossEntorpy loss of two temporal predictions
#     """

#%%
# prepare data
path=r"E:\project\images\sentinal"
img1=getBands(os.path.join(path,"170211.tif"))
img2=getBands(os.path.join(path,"190919.tif"))
data1=img1
data2=img2
indexName=["NDVI","NDWI","MSAVI","MTVI","VARI"]
labdata1=returnLabeldata(data1,indexName,reshape=False,norm=True)
labdata2=returnLabeldata(data2,indexName,reshape=False,norm=True)
data1norm=maxminNorm(data1)
data2norm=maxminNorm(data2)
data1total=np.concatenate((data1norm,labdata1),axis=2)
data2total=np.concatenate((data2norm,labdata2),axis=2)

# load label
label1=np.loadtxt(open(r"E:\project\images\sentinal\process\label1\train17.csv","r"),delimiter=',',skiprows=21,dtype=np.int)
label2=np.loadtxt(open(r"E:\project\images\sentinal\process\label1\train19.csv","r"),delimiter=',',skiprows=21,dtype=np.int)
loc1=label1[:,:2]
loc2=label2[:,:2]
train1=addY(loc1,data1total,[50,50,50])
train2=addY(loc2,data2total,[50,50,50])

# prepare mask
mask=cv2.imread(r"E:\project\images\sentinal\process\label1\mask.tif",0)
#unmask=cv2.imread(r"E:\project\images\sentinal\process\label\unmask.tif",0)
unmask=np.ones((mask.shape[0],mask.shape[1]))-mask
mask=mask.ravel()
unmask=unmask.ravel()
data1reshape=data1total.reshape((-1,data1total.shape[2]))
data2reshape=data2total.reshape((-1,data2total.shape[2]))

#初始化数据
features1=['blue1','grn1','red1','nir1','NDVI1','NDWI1','MSAVI1','MTVI1','VARI1']
features2=['blue2','grn2','red2','nir2','NDVI2','NDWI2','MSAVI2','MTVI2','VARI2']
data1df=pd.DataFrame(data1reshape,columns=features1)
data2df=pd.DataFrame(data2reshape,columns=features2)
data1df['change1']=unmask
data2df['change2']=unmask
data1df['row1']=np.array([i//772 for i in range(data1df.shape[0])])
data1df['col1']=np.array([i%772 for i in range(data1df.shape[0])])
data2df['row2']=np.array([i//772 for i in range(data2df.shape[0])])
data2df['col2']=np.array([i%772 for i in range(data2df.shape[0])])
data1df['prob1']=np.zeros((data1df.shape[0]))
data2df['prob2']=np.zeros((data2df.shape[0]))
data1df['guessClass1']=np.zeros((data1df.shape[0]))-1
data2df['guessClass2']=np.zeros((data2df.shape[0]))-1
data1df['acceptClass1']=np.zeros((data1df.shape[0]))-1
data2df['acceptClass2']=np.zeros((data2df.shape[0]))-1
datadf=pd.concat([data1df,data2df],axis=1)


# 准备数据
x1=train1[:,:-1]
y1=train1[:,-1:]
x2=train2[:,:-1]
y2=train2[:,-1:]

# load label
lab1=np.loadtxt(open(r"E:\project\images\sentinal\process\label1\test17.csv","r"),delimiter=',',skiprows=21,dtype=np.int)
lab2=np.loadtxt(open(r"E:\project\images\sentinal\process\label1\test19.csv","r"),delimiter=',',skiprows=21,dtype=np.int)
testloc1=lab1[:,:2]
testloc2=lab2[:,:2]
test1=addY(testloc1,data1total,[1000,1000,1000])
test2=addY(testloc2,data2total,[1000,1000,1000])
x_test1 = test1[:,:-1]
y_test1 = test1[:,-1]
x_test2 = test2[:,:-1]
y_test2 = test2[:,-1]


#%%
for i in range(1,4):
    resdf=runModel(datadf,x1,y1,x2,y2,x_test1,y_test1,x_test2,y_test2,c=1.0,size=5,epoch=20,labelsize=i,sampletimes=10)
    resdf.to_csv(r"E:\project\images\sentinal\process\output\res\resdf_consistency_GBDT_label"+str(i)+"_unlabel5_epoch20_c_1_0.csv",index=None)

# %%
data=[]
for i in range(1,4):
    data.append(np.loadtxt(open(r"E:\project\images\sentinal\process\output\res\resdf_unconsistency_label"+str(i)+"_unlabel"+str(i*10)+"_epoch20_c_1_0.csv",'r'),delimiter=',',skiprows=1,dtype=np.float32))
res=np.zeros((3,20,22))
for i in range(3):
    for j in range(10):
        res[i,:,:]+=data[i][j*20:(j+1)*20,:]
    res[i,:,:]/=10

# %%
epochs=np.arange(1,21,dtype=np.int)
plt.figure(figsize=(10,8))
i=0
plt.plot(epochs,res[i][:,3],color='black',marker='^',linestyle='-')
plt.plot(epochs,res[i][:,7],color='black',marker='o',linestyle='-')
plt.plot(epochs,res[i][:,11],color='black',marker='^',linestyle='-.')
plt.plot(epochs,res[i][:,15],color='black',marker='o',linestyle='-.')
plt.ylim(0.5,1)
plt.xlim(0,20)
plt.xticks(epochs)
plt.ylabel('F1-score',fontsize=20)
plt.xlabel("epoch",fontsize=20)
plt.legend(['T1 (T1&T2)','T2 (T1&T2)','T1','T2'],fontsize=16)
plt.title("labeldata "+str(i+1)+" unlabeldata 5 c 1.0 ",fontsize=20)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.savefig(r"E:\project\images\sentinal\process\output\YHLO_label"+str(i+1)+"unlabel5Epoch20c_1_0.jpg")
plt.show()

#%%
# %%
epochs=np.arange(1,21,dtype=np.int)
plt.figure(figsize=(10,8))
plt.plot(epochs,(res[0][:,18]+res[0][:,19])/2,color='black',marker='^',linestyle='-.')
plt.plot(epochs,(res[1][:,18]+res[1][:,19])/2,color='black',marker='*',linestyle='-.')
plt.plot(epochs,(res[2][:,18]+res[2][:,19])/2,color='black',marker='+',linestyle='-.')
plt.plot(epochs,(res[0][:,20]+res[0][:,21])/2,color='red',marker='^',linestyle='-.')
plt.plot(epochs,(res[1][:,20]+res[1][:,21])/2,color='red',marker='*',linestyle='-.')
plt.plot(epochs,(res[2][:,20]+res[2][:,21])/2,color='red',marker='+',linestyle='-.')
# plt.plot(epochs,res[3][:,17],color='black',marker='o',linestyle='-.')
# plt.plot(epochs,res[4][:,17],color='black',marker='v',linestyle='-.')
# plt.ylim(0.2,0.6)
plt.xlim(0,20)
plt.xticks(epochs)
plt.ylabel('Consistency Loss',fontsize=20)
plt.xlabel("epoch",fontsize=20)
plt.legend(['labeled data 1(co-train)','labeled data 2(co-train)','labeled data 3(co-train)','labeled data 1(single)','labeled data 2(single)','labeled data 3(single)'],fontsize=16)
plt.title("Consistency Loss between classifiers(unlabeled data 10 times c 1.0)",fontsize=20)
plt.tight_layout()
plt.savefig(r"E:\project\images\sentinal\process\output\label3unlabel10epoch20c1_0Coloss123.jpg")
plt.show()


# %%
