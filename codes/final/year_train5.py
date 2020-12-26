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

def multi_train(updateLocFile,train,test,data,sampletime,epochs=10,labelsize=1,unlabelsize=5,tradeoff=1):
    copy_data=data.copy()
    x_trains,y_trains=[],[]
    for i in range(k):
        x_train,y_train=selectSample(train[2*i],train[2*i+1],labelsize)
        x_trains.append(x_train)
        y_trains.append(y_train)
    # multi_train
    multi_models=[]
    for i in range(k):
        multi_models.append(RandomForestClassifier(n_estimators=50))
        multi_models[i].fit(x_trains[i],y_trains[i].ravel())
    multi_initial_result=report(test,multi_models)
    xs,ys=[],[]
    for i in range(k):
        xs.append(x_trains[i].copy())
        ys.append(y_trains[i].copy())
    for z in range(epochs):
        y_probas=[]
        for j in range(k):
            y_probas.append(multi_models[j].predict_proba(copy_data[j]))
        proba=np.zeros((copy_data[0].shape[0],n))
        for i in range(copy_data[0].shape[0]):
            for j in range(n):
                ys_proba_lst=[]
                for m in range(k):
                    ys_proba_lst.append(y_probas[m][i,j])
                proba[i,j]=fk(ys_proba_lst)
        y_pred=np.argmax(proba,axis=1)     
        y_proba=np.max(proba,axis=1)
        addids=np.array([],dtype=np.int)
        for i in range(n):
            cid=np.where(y_pred==i)[0]
            th=y_proba[cid].mean()*tradeoff
            tid=np.where(y_proba>=th)[0]
            canid=np.array(list(set(cid)&set(tid)))
            addid=np.random.choice(canid,size=unlabelsize,replace=False)
            if i==0:
                print("%d:"%z,addid)
                updateLocFile[z*unlabelsize:(z+1)*unlabelsize,0]=addid
                print(updateLocFile)
            for j in range(k):
                xs[j]=np.concatenate((xs[j],copy_data[j][addid,:]),axis=0)
                ys[j]=np.concatenate((ys[j],np.array([i]*unlabelsize).reshape((-1,1))),axis=0)
            addids=np.concatenate((addids,addid))
        for i in range(k):
            multi_models[i].fit(xs[i],ys[i].ravel())
            copy_data[i]=np.delete(copy_data[i],addids,axis=0)
    multi_final_result=report(test,multi_models)
    # self_train
    self_models=[]
    for i in range(k):
        self_models.append(RandomForestClassifier(n_estimators=50))
    for i in range(k):
        self_models[i].fit(x_trains[i],y_trains[i].ravel())
    self_initial_result=report(test,self_models)
    xs,ys=[],[]
    for i in range(k):
        xs.append(x_trains[i].copy())
        ys.append(y_trains[i].copy())
    copy_data=data.copy()
    for z in range(epochs):
        y_probas,y_preds=[],[]
        for j in range(k):
            y_probas.append(self_models[j].predict_proba(copy_data[j]).max(axis=1))
            y_preds.append(self_models[j].predict(copy_data[j]))
        addids=[]
        for j in range(k):
            addids.append(np.array([],dtype=np.int))
        for i in range(n):
            for m in range(k):
                cid=np.where(y_preds[m]==i)[0]
                th=y_probas[m][cid].mean()*tradeoff
                tid=np.where(y_probas[m]>=th)[0]
                canid=np.array(list(set(cid)&set(tid)))
                addid=np.random.choice(canid,size=unlabelsize,replace=False)
                if i==0:
                    updateLocFile[z*unlabelsize:(z+1)*unlabelsize,1+m]=addid
                addids[m]=np.concatenate((addids[m],addid))
                xs[m]=np.concatenate((xs[m],copy_data[m][addid,:]),axis=0)
                ys[m]=np.concatenate((ys[m],np.array([i]*unlabelsize).reshape((-1,1))),axis=0)
        for i in range(k):
            self_models[i].fit(xs[i],ys[i].ravel())
        for i in range(k):
            copy_data[i]=np.delete(copy_data[i],addids[i],axis=0)
    self_final_result=report(test,self_models)
    return multi_initial_result,multi_final_result,self_initial_result,self_final_result,multi_models,self_models,updateLocFile

def train_model(train,test,data,sampletimes=50,epochs=10,labelsize=1,unlabelsize=5,tradeoff=1):
    cols=[]
    for method in ['multi','self']:
        for i in range(k):
            for c in ['veg','shadow','water','road','building','total']:
                cols.append(method+'_'+c+str(i))
    resdf=pd.DataFrame(np.zeros((sampletimes*2,k*12)),columns=cols)
    for sampletime in range(sampletimes):
        print("sampletime:",sampletime)
        updateLocFile=np.zeros((unlabelsize*epochs,k+1),dtype=np.int64)
        multi_initial_result,multi_final_result,self_initial_result,self_final_result,multi_models,self_models,updateLocFile=multi_train(updateLocFile,train,test,data,sampletime,epochs,labelsize,unlabelsize,tradeoff)
        np.savetxt(locpath+"\label{}_unlabel{}_sampletime{}_updataloc.csv".format(labelsize,unlabelsize,sampletime),updateLocFile,delimiter=',')
        resdf.iloc[sampletime*2,:k*6]=multi_initial_result.ravel()
        resdf.iloc[sampletime*2+1,:k*6]=multi_final_result.ravel()
        resdf.iloc[sampletime*2,k*6:]=self_initial_result.ravel()
        resdf.iloc[sampletime*2+1,k*6:]=self_final_result.ravel()
        print("initial: ")
        for i in range(k):
            print("multi:%d:%.4f,self:%d:%.4f"%(i+1,multi_initial_result[i,5],i+1,self_initial_result[i,5]))
        print("final:")
        for i in range(k):
            print("multi:%d:%.4f,self:%d:%.4f"%(i+1,multi_final_result[i,5],i+1,self_final_result[i,5]))
        multi_ypreds,self_ypreds=[],[]
        for i in range(k):
            multi_ypreds.append(multi_models[i].predict(datareshapes[i]).astype(np.uint8).reshape(653,772))
            self_ypreds.append(self_models[i].predict(datareshapes[i]).astype(np.uint8).reshape(653,772))
            cv2.imwrite(os.path.join(map_dir,"multi_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_"+str(tradeoff*100)+"class"+str(i)+"_"+str(sampletime)+".tif"),multi_ypreds[i])
            cv2.imwrite(os.path.join(map_dir,"self_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_"+str(tradeoff*100)+"class"+str(i)+"_"+str(sampletime)+".tif"),self_ypreds[i])
    resdf.to_csv(os.path.join(csv_path,"multi_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"epochs_"+str(epochs)+"sampletimes_"+str(sampletimes)+"c_"+str(int(tradeoff*100))+".csv"),index=None)

#%%
# # prepare data
# k=int(input("请输入时相数量：（如4）\n"))
# image_path=input("请输入原始图像文件夹：（如G:\project\images\sentinal）\n")
# image_names=["170211.tif","170402.tif","171009.tif","171218.tif","180313.tif","180611.tif","180909.tif","190122.tif"]
# label_names=["train170211.csv","train170402.csv","train171009.csv","train171218.csv","train180313.csv","train180611.csv","train180909.csv","train190122.csv"]
# image_name_lst=[]
# for i in range(k):
#     image_name=int(input("请输入原始图像{}序号：（如1）\n".format(i)))
#     image_name_lst.append(image_names[image_name-1])
# label_path=input("请输入标签路径：（如G:\project\images\sentinal\process\labels）\n")
# label_name_lst=[]
# for i in range(k):
#     label_name=int(input("请输入标签{}序号：（如1）\n".format(i)))
#     label_name_lst.append(label_names[label_name-1])
# map_dir=input("请输入分类结果图文件夹：（如G:\project\images\sentinal\process\output\map\17）\n")
# csv_path=input("请输入输出csv文件夹：（如G:\project\images\sentinal\process\output\res\17）\n")
# epochs=int(input("请输入迭代次数：（如10）\n"))
# sampletimes=int(input("请输入独立重复采样次数：（如50）\n"))
# labelsize_min,labelsize_max,labelsize_step=eval(input("请输入有标记样本范围：（如1,5,2）\n"))
# unlabelsize_min,unlabelsize_max,unlabelsize_step=eval(input("请输入无标记样本范围：（如5,50,5）\n"))
# tradeoff_min,tradeoff_max,tradeoff_step=eval(input("请输入tradeoff范围：（如0.8,1.2,0.05）\n"))
# locpath=input("请输入更新位置坐标文件存储地址\n")
# prepare data
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
# load label
labels=[]
locs=[]
trains=[]
x_trains,x_tests,y_trains,y_tests=[],[],[],[]
for i in range(k):
    labels.append(np.loadtxt(open(os.path.join(label_path,label_name_lst[i]),"r"),delimiter=',',skiprows=25,dtype=np.int))
    locs.append(labels[i][:,:2])
    trains.append(addY(locs[i],datatotals[i],[1000]*n))
    x_train,x_test,y_train,y_test=random_train_test_split(trains[i],size=1000)
    x_trains.append(x_train)
    x_tests.append(x_test)
    y_trains.append(y_train)
    y_tests.append(y_test)
# %%
train=[]
test=[]
data=[]
for i in range(k):
    train.append(x_trains[i])
    train.append(y_trains[i])
    test.append(x_tests[i])
    test.append(y_tests[i])
    data.append(datareshapes[i])
for labelsize in range(labelsize_min,labelsize_max+labelsize_step,labelsize_step):
    print("labelsize:",labelsize)
    for unlabelsize in range(unlabelsize_min,unlabelsize_max+unlabelsize_step,unlabelsize_step):
        print("unlabelsize:",unlabelsize)
        for tradeoff in range(int(tradeoff_min*100),int((tradeoff_max+tradeoff_step)*100),int(tradeoff_step*100)):
            tradeoff/=100
            print("tradeoff:",tradeoff)
            train_model(train,test,data,sampletimes,epochs,labelsize,unlabelsize,tradeoff)
# %%
