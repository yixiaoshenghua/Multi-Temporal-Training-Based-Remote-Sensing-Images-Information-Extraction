#%%
#from lib import *
#%%
import gdal
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn import svm
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.metrics import log_loss,confusion_matrix,plot_confusion_matrix,f1_score,classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,BaggingClassifier
import scipy.stats

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
    y=np.array([0]*nums[0]+[1]*nums[1]+[2]*nums[2]+[3]*nums[3]+[4]*nums[4]).reshape((sum(nums),1))
    x=np.zeros((sum(nums),data.shape[2]))
    for i,yx in enumerate(loc):
        x[i,:]=data[yx[1],yx[0],:]
    train=np.concatenate((x,y),axis=1)
    return train

def cotrain(datadf,x1,y1,x2,y2,model1,model2,c,size=5,selflearn=False):
    # 候选预测数据概率和标签
    datadf['prob1']=model1.predict_proba(datadf[features1]).max(axis=1)
    datadf['prob2']=model2.predict_proba(datadf[features2]).max(axis=1)
    datadf['guessClass1']=model1.predict(datadf[features1])
    datadf['guessClass2']=model2.predict(datadf[features2])
    sizes=[size]*n
    if selflearn==False:
        # 找到guessclass一致区域的数据
        datadf_candidate=datadf[datadf['change1']==0][datadf['guessClass1']==datadf['guessClass2']]
        # 标签一致区域找到置信度最高的区域
        class_candidate=[datadf_candidate[datadf_candidate['guessClass1']==i] for i in range(n)]
        # 找到两期置信度都高的区域更新训练集
        for i in range(n):
            # tradeoff
            th1=c*class_candidate[i]['prob1'].mean()
            th2=c*class_candidate[i]['prob2'].mean()
            index1=np.array(class_candidate[i][class_candidate[i]['prob1']>th1].index)
            index2=np.array(class_candidate[i][class_candidate[i]['prob2']>th2].index)
            index=np.array([x for x in index1 if x in index2])
            try:
                sample1=np.random.randint(low=0,high=int(index.shape[0]),size=int(sizes[i]))
                sample2=np.random.randint(low=0,high=int(index.shape[0]),size=int(sizes[i]))
            except ValueError:
                return 0
            can1=index1[sample1]
            can2=index2[sample2]
            x1=np.concatenate((x1,class_candidate[i].loc[can1,features1]),axis=0)
            x2=np.concatenate((x2,class_candidate[i].loc[can2,features2]),axis=0)
            y1=np.concatenate((y1,class_candidate[i].loc[can1,['guessClass1']].values.reshape((-1,1))),axis=0)
            y2=np.concatenate((y2,class_candidate[i].loc[can2,['guessClass2']].values.reshape((-1,1))),axis=0)
    else:
        class_candidate1=[datadf[datadf['guessClass1']==i] for i in range(n)]
        class_candidate2=[datadf[datadf['guessClass2']==i] for i in range(n)]
        for i in range(n):
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

def run(datadf,x1,y1,x2,y2,model1,model2,x_test1,y_test1,x_test2,y_test2,c,size=5,epochs=10,selflearn=False,curepoch=0):
    if selflearn==False:
        for i in range(epochs):
            model1.fit(x1,y1.ravel())
            model2.fit(x2,y2.ravel())
            y_pre1 = model1.predict(x_test1)
            y_pre2 = model2.predict(x_test2)
            y1_pred=model1.predict(datadf.loc[:,features1].values)
            y2_pred=model2.predict(datadf.loc[:,features2].values)
            if i==0:
                ii=curepoch*2
                resdf.at[ii,'coPDC']=calcPDC(datadf,y1_pred,y2_pred)
                res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','shadow','water','road','building'],
                                    output_dict=True)
                res2=classification_report(y_test2,y_pre2,
                                    target_names=['veg','shadow','water','road','building'],
                                    output_dict=True)
                resdf.iat[ii,0],resdf.iat[ii,1],resdf.iat[ii,2],resdf.iat[ii,3],resdf.iat[ii,4],resdf.iat[ii,5]=res1['veg']['f1-score'],res1['shadow']['f1-score'],res1['water']['f1-score'],res1['road']['f1-score'],res1['building']['f1-score'],res1['weighted avg']['f1-score']
                resdf.iat[ii,6],resdf.iat[ii,7],resdf.iat[ii,8],resdf.iat[ii,9],resdf.iat[ii,10],resdf.iat[ii,11]=res2['veg']['f1-score'],res2['shadow']['f1-score'],res2['water']['f1-score'],res2['road']['f1-score'],res2['building']['f1-score'],res2['weighted avg']['f1-score']
                print("score: f1:%.5f f2:%.5f"%(res1['weighted avg']['f1-score'],res2['weighted avg']['f1-score']))
            try:
                x1,y1,x2,y2,model1,model2=cotrain(datadf,x1,y1,x2,y2,model1,model2,c,size,selflearn)
            except TypeError:
                return 0
            if i==epochs-1:
                ii=curepoch*2+1
                resdf.at[ii,'coPDC']=calcPDC(datadf,y1_pred,y2_pred)
                res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','shadow','water','road','building'],
                                    output_dict=True)
                res2=classification_report(y_test2,y_pre2,
                                    target_names=['veg','shadow','water','road','building'],
                                    output_dict=True)
                resdf.iat[ii,0],resdf.iat[ii,1],resdf.iat[ii,2],resdf.iat[ii,3],resdf.iat[ii,4],resdf.iat[ii,5]=res1['veg']['f1-score'],res1['shadow']['f1-score'],res1['water']['f1-score'],res1['road']['f1-score'],res1['building']['f1-score'],res1['weighted avg']['f1-score']
                resdf.iat[ii,6],resdf.iat[ii,7],resdf.iat[ii,8],resdf.iat[ii,9],resdf.iat[ii,10],resdf.iat[ii,11]=res2['veg']['f1-score'],res2['shadow']['f1-score'],res2['water']['f1-score'],res2['road']['f1-score'],res2['building']['f1-score'],res2['weighted avg']['f1-score']
                print("score: f1:%.5f f2:%.5f"%(res1['weighted avg']['f1-score'],res2['weighted avg']['f1-score']))
        return model1,model2
    else:
        for i in range(epochs):
            model1.fit(x1,y1.ravel())
            model2.fit(x2,y2.ravel())
            y_pre1 = model1.predict(x_test1)
            y_pre2 = model2.predict(x_test2)
            y1_pred=model1.predict(datadf.loc[:,features1].values)
            y2_pred=model2.predict(datadf.loc[:,features2].values)
            if i==0:
                ii=curepoch*2
                resdf.at[ii,'sePDC']=calcPDC(datadf,y1_pred,y2_pred)
                res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','shadow','water','road','building'],
                                    output_dict=True)
                res2=classification_report(y_test2,y_pre2,
                                    target_names=['veg','shadow','water','road','building'],
                                    output_dict=True)
                resdf.iat[ii,12],resdf.iat[ii,13],resdf.iat[ii,14],resdf.iat[ii,15],resdf.iat[ii,16],resdf.iat[ii,17]=res1['veg']['f1-score'],res1['shadow']['f1-score'],res1['water']['f1-score'],res1['road']['f1-score'],res1['building']['f1-score'],res1['weighted avg']['f1-score']
                resdf.iat[ii,18],resdf.iat[ii,19],resdf.iat[ii,20],resdf.iat[ii,21],resdf.iat[ii,22],resdf.iat[ii,23]=res2['veg']['f1-score'],res2['shadow']['f1-score'],res2['water']['f1-score'],res2['road']['f1-score'],res2['building']['f1-score'],res2['weighted avg']['f1-score']
                print("score: f1:%.5f f2:%.5f"%(res1['weighted avg']['f1-score'],res2['weighted avg']['f1-score']))
            x1,y1,x2,y2,model1,model2=cotrain(datadf,x1,y1,x2,y2,model1,model2,c,size,selflearn)
            if i==epochs-1:
                ii=curepoch*2+1
                resdf.at[ii,'sePDC']=calcPDC(datadf,y1_pred,y2_pred)
                res1=classification_report(y_test1,y_pre1,
                                    target_names=['veg','shadow','water','road','building'],
                                    output_dict=True)
                res2=classification_report(y_test2,y_pre2,
                                    target_names=['veg','shadow','water','road','building'],
                                    output_dict=True)
                resdf.iat[ii,12],resdf.iat[ii,13],resdf.iat[ii,14],resdf.iat[ii,15],resdf.iat[ii,16],resdf.iat[ii,17]=res1['veg']['f1-score'],res1['shadow']['f1-score'],res1['water']['f1-score'],res1['road']['f1-score'],res1['building']['f1-score'],res1['weighted avg']['f1-score']
                resdf.iat[ii,18],resdf.iat[ii,19],resdf.iat[ii,20],resdf.iat[ii,21],resdf.iat[ii,22],resdf.iat[ii,23]=res2['veg']['f1-score'],res2['shadow']['f1-score'],res2['water']['f1-score'],res2['road']['f1-score'],res2['building']['f1-score'],res2['weighted avg']['f1-score']
                print("score: f1:%.5f f2:%.5f"%(res1['weighted avg']['f1-score'],res2['weighted avg']['f1-score']))
        return model1,model2

def runModel(datadf,x1,y1,x2,y2,x_test1,y_test1,x_test2,y_test2,c=1.0,unlabelsize=5,labelsize=1,epochs=10,sampletimes=20):
    # 构造model训练
    i=0
    while i<sampletimes:
        x1_train,y1_train=selectSample(x1,y1,labelsize)
        x2_train,y2_train=selectSample(x2,y2,labelsize)
        print("co-train %d sampletime"%i)
        model1=RandomForestClassifier(n_estimators=50)
        model2=RandomForestClassifier(n_estimators=50)
        x_train1=x1_train.copy()
        x_train2=x2_train.copy()
        y_train1=y1_train.copy()
        y_train2=y2_train.copy()
        try:
            model1,model2=run(datadf,x_train1,y_train1,x_train2,y_train2,model1,model2,x_test1,y_test1,x_test2,y_test2,c=c,size=unlabelsize,epochs=epochs,selflearn=False,curepoch=i)
        except TypeError:
            continue
        y_pred1=model1.predict(data1reshape).astype(np.uint8).reshape((653,772))
        cv2.imwrite(r"G:\project\images\sentinal\process\output\map\161103_170211\co_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"class1_"+str(i)+".tif",y_pred1)
        y_pred2=model2.predict(data2reshape).astype(np.uint8).reshape((653,772))
        cv2.imwrite(r"G:\project\images\sentinal\process\output\map\161103_170211\co_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"class2_"+str(i)+".tif",y_pred2)        
        print("self-train %d sampletime"%i)
        model3=RandomForestClassifier(n_estimators=50)
        model4=RandomForestClassifier(n_estimators=50)
        x_train3=x1_train.copy()
        x_train4=x2_train.copy()
        y_train3=y1_train.copy()
        y_train4=y2_train.copy()
        model3,model4=run(datadf,x_train3,y_train3,x_train4,y_train4,model3,model4,x_test1,y_test1,x_test2,y_test2,c=c,size=unlabelsize,epochs=epochs,selflearn=True,curepoch=i)
        y_pred3=model3.predict(data1reshape).astype(np.uint8).reshape((653,772))
        cv2.imwrite(r"G:\project\images\sentinal\process\output\map\161103_170211\self_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"class1_"+str(i)+".tif",y_pred3)
        y_pred4=model2.predict(data2reshape).astype(np.uint8).reshape((653,772))
        cv2.imwrite(r"G:\project\images\sentinal\process\output\map\161103_170211\self_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"class2_"+str(i)+".tif",y_pred4)
        i+=1

def selectSample(x,y,sample):
    m=np.zeros((sample*n),dtype=np.int)
    for i in range(n):
        m[i*sample:(i+1)*sample]=np.random.randint(i*200,(i+1)*200,size=sample)
    return x[m,:],y[m,:]    

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

#%%
# prepare data
n=5
path=r"G:\project\images\sentinal"
img1=getBands(os.path.join(path,"161103.tif"))
img2=getBands(os.path.join(path,"170211.tif"))
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
label1=np.loadtxt(open(r"G:\project\images\sentinal\process\labels\train161103.csv","r"),delimiter=',',skiprows=25,dtype=np.int)
label2=np.loadtxt(open(r"G:\project\images\sentinal\process\labels\train170211.csv","r"),delimiter=',',skiprows=25,dtype=np.int)
loc1=label1[:,:2]
loc2=label2[:,:2]
train1=addY(loc1,data1total,[1000]*n)
train2=addY(loc2,data2total,[1000]*n)

unmask=returnUnlabeledData(data1total.reshape((-1,9)),data2total.reshape((-1,9)))
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

x_train1,x_test1,y_train1,y_test1=random_train_test_split(train1,size=1000)
x_train2,x_test2,y_train2,y_test2=random_train_test_split(train2,size=1000)


#%%
for labelsize in range(1,5,2):
    print("labelsize:",labelsize)
    for unlabelsize in range(30,55,5):
        print("unlabelsize:",unlabelsize)
        epochs=10
        sampletimes=20
        resdf=pd.DataFrame(np.zeros((sampletimes*2,26)),
                    columns=['17co_veg','17co_shadow','17co_water','17co_road','17co_building','17co_total',
                            '19co_veg','19co_shadow','19co_water','19co_road','19co_building','19co_total',
                            '17se_veg','17se_shadow','17se_water','17se_road','17se_building','17se_total',
                            '19se_veg','19se_shadow','19se_water','19se_road','19se_building','19se_total',
                            'coPDC','sePDC'])
        runModel(datadf,x_train1,y_train1,x_train2,y_train2,x_test1,y_test1,x_test2,y_test2,c=1.0,unlabelsize=unlabelsize,labelsize=labelsize,epochs=epochs,sampletimes=sampletimes)
        resdf.to_csv(r"G:\project\images\sentinal\process\output\res\161103_170211_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"epochs_"+str(epochs)+"sampletimes_"+str(sampletimes)+"c_1_0.csv",index=None)

# %%
data=[]
for unlabelsize in range(5,55,5):
    data.append(np.loadtxt(open(r"G:\project\images\sentinal\process\output\result\161103_170211\label_1unlabel_"+str(unlabelsize)+"epochs_10sampletimes_50c_100.csv",'r'),delimiter=',',skiprows=1,dtype=np.float32))
res=np.zeros((20,26))
for i in range(10):
    for j in range(50):
        res[i,:]+=data[i][j*2,:]
        res[10+i,:]+=data[i][j*2+1,:]
res/=50

# %%
epochs=np.arange(5,55,5,dtype=np.int)
plt.figure(figsize=(10,8))

plt.plot(epochs,res[10:,0],color='green',marker='^',linestyle='-')
plt.plot(epochs,res[10:,6],color='green',marker='o',linestyle='-')
# plt.plot(epochs,res[10:,17],color='black',marker='^',linestyle='-')
# plt.plot(epochs,res[10:,23],color='black',marker='o',linestyle='-')
plt.plot(epochs,res[:10,0],color='green',marker='^',linestyle='-.')
plt.plot(epochs,res[:10,6],color='green',marker='o',linestyle='-.')
# plt.plot(epochs,res[:10,17],color='black',marker='^',linestyle='-.')
# plt.plot(epochs,res[:10,23],color='black',marker='o',linestyle='-.')
plt.ylim(0.8,1)
plt.xlim(0,50)
plt.xticks(epochs)
plt.ylabel('F1-score',fontsize=20)
plt.xlabel("unlabeled dataSize",fontsize=20)
#plt.legend(['T1 (T1&T2) final','T2 (T1&T2) final','T1 final','T2 final','T1 (T1&T2) initial','T2 (T1&T2) initial','T1 initial','T2 initial'],fontsize=16)
plt.legend(['T1 final','T2 final','T1 initial','T2 initial'],fontsize=16)
#plt.legend(['T1 (T1&T2) final','T2 (T1&T2) final','T1 (T1&T2) initial','T2 (T1&T2) initial'],fontsize=16)
plt.title("F1-score of average (labeled dataSize="+str(i+1)+", c=1.0)",fontsize=20)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
#plt.savefig(r"G:\project\images\sentinal\process\output\plot\161103_170211\labelsize"+str(i+1)+"c_1_0_ave_self.jpg")
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
print((res[9:,:6]+res[9:,6:12]-(res[:9,:6]+res[:9,6:12]))/(res[:9,:6]+res[:9,6:12]))
print((res[:9,:6]+res[:9,6:12])/2)
print((res[9:,:6]+res[9:,6:12])/2)
print((res[9:,12:18]+res[9:,18:24]-(res[:9,12:18]+res[:9,18:24]))/(res[:9,12:18]+res[:9,18:24]))
print((res[:9,12:18]+res[:9,18:24])/2)
print((res[9:,12:18]+res[9:,18:24])/2)

# %%
# %%
x1_train,y1_train=selectSample(x_train1,y_train1,1)
x2_train,y2_train=selectSample(x_train2,y_train2,1)
from sklearn import svm
clf=svm.SVC(kernel='rbf',probability=True)
clf.fit(x1_train,y1_train.ravel())
# %%
y_proba1=clf.predict_proba(x_test1)
y_pred1=clf.predict(x_test1)
# %%
classification_report(y_test1,y_pred1,
                                    target_names=['veg','shadow','water','road','building'],
                                    output_dict=True)
# %%
clfbag=BaggingClassifier(svm.SVC(kernel='rbf',probability=True),max_samples=1, max_features=0.5,oob_score=True,n_jobs=-1)
clfbag.fit(x1_train,y1_train.ravel())
# %%
y_proba1=clfbag.predict_proba(x_test1)
y_pred1=clfbag.predict(x_test1)
classification_report(y_test1,y_pred1,
                                    target_names=['veg','shadow','water','road','building'],
                                    output_dict=True)
# %%


