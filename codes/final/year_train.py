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

def multi_train(train,test,data,sampletime,epochs=10,labelsize=1,unlabelsize=5,tradeoff=1):
    x_train1,y_train1,x_train2,y_train2,x_train3,y_train3,x_train4,y_train4=train
    data1,data2,data3,data4=data.copy()
    x1_train,y1_train=selectSample(x_train1,y_train1,labelsize)
    x2_train,y2_train=selectSample(x_train2,y_train2,labelsize)
    x3_train,y3_train=selectSample(x_train3,y_train3,labelsize)
    x4_train,y4_train=selectSample(x_train4,y_train4,labelsize)
    # multi_train
    multi_model1,multi_model2,multi_model3,multi_model4=RandomForestClassifier(n_estimators=50),RandomForestClassifier(n_estimators=50),RandomForestClassifier(n_estimators=50),RandomForestClassifier(n_estimators=50)
    multi_model1.fit(x1_train,y1_train.ravel())
    multi_model2.fit(x2_train,y2_train.ravel())
    multi_model3.fit(x3_train,y3_train.ravel())
    multi_model4.fit(x4_train,y4_train.ravel())
    multi_initial_result=report(test,multi_model1,multi_model2,multi_model3,multi_model4)
    x1,x2,x3,x4=x1_train.copy(),x2_train.copy(),x3_train.copy(),x4_train.copy()
    y1,y2,y3,y4=y1_train.copy(),y2_train.copy(),y3_train.copy(),y4_train.copy()
    for i in range(epochs):
        y1_proba=multi_model1.predict_proba(data1)
        y2_proba=multi_model2.predict_proba(data2)
        y3_proba=multi_model3.predict_proba(data3)
        y4_proba=multi_model4.predict_proba(data4)
        proba=np.zeros((data1.shape[0],n))
        for i in range(data1.shape[0]):
            for j in range(n):
                proba[i,j]=f4(y1_proba[i,j],y2_proba[i,j],y3_proba[i,j],y4_proba[i,j])
        y_pred=np.argmax(proba,axis=1)     
        y_proba=np.max(proba,axis=1)
        addids=np.array([])
        for i in range(n):
            cid=np.where(y_pred==i)[0]
            th=y_proba[cid].mean()*tradeoff
            tid=np.where(y_proba>=th)[0]
            canid=np.array(list(set(cid)&set(tid)))
            addid=np.random.choice(canid,size=unlabelsize,replace=False)
            x1=np.concatenate((x1,data1[addid,:]),axis=0)
            x2=np.concatenate((x2,data2[addid,:]),axis=0)
            x3=np.concatenate((x3,data3[addid,:]),axis=0)
            x4=np.concatenate((x4,data4[addid,:]),axis=0)
            y1=np.concatenate((y1,np.array([i]*unlabelsize).reshape((-1,1))),axis=0)
            y2=np.concatenate((y2,np.array([i]*unlabelsize).reshape((-1,1))),axis=0)
            y3=np.concatenate((y3,np.array([i]*unlabelsize).reshape((-1,1))),axis=0)
            y4=np.concatenate((y4,np.array([i]*unlabelsize).reshape((-1,1))),axis=0)
            addids=np.concatenate((addids,addid))
        multi_model1.fit(x1,y1.ravel())
        multi_model2.fit(x2,y2.ravel())
        multi_model3.fit(x3,y3.ravel())
        multi_model4.fit(x4,y4.ravel())
        data1=np.delete(data1,addids,axis=0)
        data2=np.delete(data2,addids,axis=0)
        data3=np.delete(data3,addids,axis=0)
        data4=np.delete(data4,addids,axis=0)
    multi_final_result=report(test,multi_model1,multi_model2,multi_model3,multi_model4)
    # self_train
    self_model1,self_model2,self_model3,self_model4=RandomForestClassifier(n_estimators=50),RandomForestClassifier(n_estimators=50),RandomForestClassifier(n_estimators=50),RandomForestClassifier(n_estimators=50)
    self_model1.fit(x1_train,y1_train.ravel())
    self_model2.fit(x2_train,y2_train.ravel())
    self_model3.fit(x3_train,y3_train.ravel())
    self_model4.fit(x4_train,y4_train.ravel())
    self_initial_result=report(test,self_model1,self_model2,self_model3,self_model4)
    x1,x2,x3,x4=x1_train.copy(),x2_train.copy(),x3_train.copy(),x4_train.copy()
    y1,y2,y3,y4=y1_train.copy(),y2_train.copy(),y3_train.copy(),y4_train.copy()
    data1,data2,data3,data4=data.copy()
    for i in range(epochs):
        y1_proba=self_model1.predict_proba(data1).max(axis=1)
        y2_proba=self_model2.predict_proba(data2).max(axis=1)
        y3_proba=self_model3.predict_proba(data3).max(axis=1)
        y4_proba=self_model4.predict_proba(data4).max(axis=1)
        y1_pred=self_model1.predict(data1)
        y2_pred=self_model2.predict(data2)
        y3_pred=self_model3.predict(data3)
        y4_pred=self_model4.predict(data4)
        addids1,addids2,addids3,addids4=np.array([]),np.array([]),np.array([]),np.array([])
        for i in range(n):
            cid1=np.where(y1_pred==i)[0]
            th1=y1_proba[cid1].mean()*tradeoff
            tid1=np.where(y1_proba>=th1)[0]
            canid1=np.array(list(set(cid1)&set(tid1)))
            addid1=np.random.choice(canid1,size=unlabelsize,replace=False)
            addids1=np.concatenate((addids1,addid1))
            cid2=np.where(y2_pred==i)[0]
            th2=y2_proba[cid2].mean()*tradeoff
            tid2=np.where(y2_proba>=th2)[0]
            canid2=np.array(list(set(cid2)&set(tid2)))
            addid2=np.random.choice(canid2,size=unlabelsize,replace=False)
            addids2=np.concatenate((addids2,addid2))
            cid3=np.where(y3_pred==i)[0]
            th3=y3_proba[cid3].mean()*tradeoff
            tid3=np.where(y3_proba>=th3)[0]
            canid3=np.array(list(set(cid3)&set(tid3)))
            addid3=np.random.choice(canid3,size=unlabelsize,replace=False)
            addids3=np.concatenate((addids3,addid3))
            cid4=np.where(y4_pred==i)[0]
            th4=y4_proba[cid4].mean()*tradeoff
            tid4=np.where(y4_proba>=th4)[0]
            canid4=np.array(list(set(cid4)&set(tid4)))
            addid4=np.random.choice(canid4,size=unlabelsize,replace=False)
            addids4=np.concatenate((addids4,addid4))
            x1=np.concatenate((x1,data1[addid1,:]),axis=0)
            x2=np.concatenate((x2,data2[addid2,:]),axis=0)
            x3=np.concatenate((x3,data3[addid3,:]),axis=0)
            x4=np.concatenate((x4,data4[addid4,:]),axis=0)
            y1=np.concatenate((y1,np.array([i]*unlabelsize).reshape((-1,1))),axis=0)
            y2=np.concatenate((y2,np.array([i]*unlabelsize).reshape((-1,1))),axis=0)
            y3=np.concatenate((y3,np.array([i]*unlabelsize).reshape((-1,1))),axis=0)
            y4=np.concatenate((y4,np.array([i]*unlabelsize).reshape((-1,1))),axis=0)
        self_model1.fit(x1,y1.ravel())
        self_model2.fit(x2,y2.ravel())
        self_model3.fit(x3,y3.ravel())
        self_model4.fit(x4,y4.ravel())
        data1=np.delete(data1,addids1,axis=0)
        data2=np.delete(data2,addids2,axis=0)
        data3=np.delete(data3,addids3,axis=0)
        data4=np.delete(data4,addids4,axis=0)
    self_final_result=report(test,self_model1,self_model2,self_model3,self_model4)
    return multi_initial_result,multi_final_result,self_initial_result,self_final_result,multi_model1,multi_model2,multi_model3,multi_model4,self_model1,self_model2,self_model3,self_model4

def train_model(train,test,data,sampletimes=50,epochs=10,labelsize=1,unlabelsize=5,tradeoff=1):
    resdf=pd.DataFrame(np.zeros((sampletimes*2,48)),
                    columns=['multi_veg1','multi_shadow1','multi_water1','multi_road1','multi_building1','multi_total1',
                            'multi_veg2','multi_shadow2','multi_water2','multi_road2','multi_building2','multi_total2',
                            'multi_veg3','multi_shadow3','multi_water3','multi_road3','multi_building3','multi_total3',
                            'multi_veg4','multi_shadow4','multi_water4','multi_road4','multi_building4','multi_total4',
                            'self_veg1','self_shadow1','self_water1','self_road1','self_building1','self_total1',
                            'self_veg2','self_shadow2','self_water2','self_road2','self_building2','self_total2',
                            'self_veg3','self_shadow3','self_water3','self_road3','self_building3','self_total3',
                            'self_veg4','self_shadow4','self_water4','self_road4','self_building4','self_total4'
                            ])
    for sampletime in range(sampletimes):
        print("sampletime:",sampletime)
        multi_initial_result,multi_final_result,self_initial_result,self_final_result,multi_model1,multi_model2,multi_model3,multi_model4,self_model1,self_model2,self_model3,self_model4=multi_train(train,test,data,sampletime,epochs,labelsize,unlabelsize,tradeoff)
        resdf.iloc[sampletime*2,:24]=multi_initial_result.ravel()
        resdf.iloc[sampletime*2+1,:24]=multi_final_result.ravel()
        resdf.iloc[sampletime*2,24:]=self_initial_result.ravel()
        resdf.iloc[sampletime*2+1,24:]=self_final_result.ravel()
        print("initial: multi:1:%.4f,2:%.4f,3:%.4f,4:%.4f,self:1:%.4f,2:%.4f,3:%.4f,4:%.4f"%(multi_initial_result[0,5],multi_initial_result[1,5],multi_initial_result[2,5],multi_initial_result[3,5],self_initial_result[0,5],self_initial_result[1,5],self_initial_result[2,5],self_initial_result[3,5]))
        print("final: multi:1:%.4f,2:%.4f,3:%.4f,4:%.4f,self:1:%.4f,2:%.4f,3:%.4f,4:%.4f"%(multi_final_result[0,5],multi_final_result[1,5],multi_final_result[2,5],multi_final_result[3,5],self_final_result[0,5],self_final_result[1,5],self_final_result[2,5],self_final_result[3,5]))
        multi_ypred1=multi_model1.predict(data1reshape).astype(np.uint8).reshape((653,772))
        multi_ypred2=multi_model2.predict(data2reshape).astype(np.uint8).reshape((653,772))
        multi_ypred3=multi_model3.predict(data3reshape).astype(np.uint8).reshape((653,772))
        multi_ypred4=multi_model4.predict(data4reshape).astype(np.uint8).reshape((653,772))
        self_ypred1=self_model1.predict(data1reshape).astype(np.uint8).reshape((653,772))
        self_ypred2=self_model2.predict(data2reshape).astype(np.uint8).reshape((653,772))
        self_ypred3=self_model3.predict(data3reshape).astype(np.uint8).reshape((653,772))
        self_ypred4=self_model4.predict(data4reshape).astype(np.uint8).reshape((653,772))
        cv2.imwrite(os.path.join(map_dir,"multi_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_"+str(tradeoff*100)+"class1_"+str(sampletime)+".tif"),multi_ypred1)
        cv2.imwrite(os.path.join(map_dir,"multi_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_"+str(tradeoff*100)+"class2_"+str(sampletime)+".tif"),multi_ypred2)
        cv2.imwrite(os.path.join(map_dir,"multi_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_"+str(tradeoff*100)+"class3_"+str(sampletime)+".tif"),multi_ypred3)
        cv2.imwrite(os.path.join(map_dir,"multi_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_"+str(tradeoff*100)+"class4_"+str(sampletime)+".tif"),multi_ypred4)
        cv2.imwrite(os.path.join(map_dir,"self_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_"+str(tradeoff*100)+"class1_"+str(sampletime)+".tif"),self_ypred1)
        cv2.imwrite(os.path.join(map_dir,"self_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_"+str(tradeoff*100)+"class2_"+str(sampletime)+".tif"),self_ypred2)
        cv2.imwrite(os.path.join(map_dir,"self_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_"+str(tradeoff*100)+"class3_"+str(sampletime)+".tif"),self_ypred3)
        cv2.imwrite(os.path.join(map_dir,"self_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"c_"+str(tradeoff*100)+"class4_"+str(sampletime)+".tif"),self_ypred4)
    resdf.to_csv(os.path.join(csv_path,"multi_label_"+str(labelsize)+"unlabel_"+str(unlabelsize)+"epochs_"+str(epochs)+"sampletimes_"+str(sampletimes)+"c_"+str(tradeoff*100)+".csv"),index=None)
#%%
# prepare data
image_path=input("请输入原始图像文件夹：（如G:\project\images\sentinal）\n")
image1_name=input("请输入原始图像1名称：（如170211.tif）\n")
image2_name=input("请输入原始图像2名称：（如170402.tif）\n")
image3_name=input("请输入原始图像3名称：（如171009.tif）\n")
image4_name=input("请输入原始图像4名称：（如171218.tif）\n")
label_path=input("请输入标签路径：（如G:\project\images\sentinal\process\labels）\n")
label1_name=input("请输入标签1名称：（如train170211.csv）\n")
label2_name=input("请输入标签2名称：（如train170402.csv）\n")
label3_name=input("请输入标签3名称：（如train171009.csv）\n")
label4_name=input("请输入标签4名称：（如train171218.csv）\n")
map_dir=input("请输入分类结果图文件夹：（如G:\project\images\sentinal\process\output\map\17）\n")
csv_path=input("请输入输出csv文件夹：（如G:\project\images\sentinal\process\output\res\17）\n")
epochs=int(input("请输入迭代次数：（如10）\n"))
sampletimes=int(input("请输入独立重复采样次数：（如50）\n"))
labelsize_min,labelsize_max,labelsize_step=eval(input("请输入有标记样本范围：（如1,5,2）\n"))
unlabelsize_min,unlabelsize_max,unlabelsize_step=eval(input("请输入无标记样本范围：（如5,50,5）\n"))
tradeoff=int(input("请输入trade-off值：（如1.0）\n"))
n=5
img1=getBands(os.path.join(image_path,image1_name))
img2=getBands(os.path.join(image_path,image2_name))
img3=getBands(os.path.join(image_path,image3_name))
img4=getBands(os.path.join(image_path,image4_name))
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
# load label
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
for labelsize in range(labelsize_min,labelsize_max+labelsize_step,labelsize_step):
    print("labelsize:",labelsize)
    for unlabelsize in range(unlabelsize_min,unlabelsize_max+unlabelsize_step,unlabelsize_step):
        print("unlabelsize:",unlabelsize)
        train_model(train,test,data,sampletimes,epochs,labelsize,unlabelsize,tradeoff)
# %%
