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
from tqdm import tqdm
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

#%%
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier

"""
求解核矩阵K
    ker:求解核函数的方法
    X1:源域数据的特征矩阵
    X2:目标域数据的特征矩阵
    gamma:当核函数方法选择rbf时，的参数
"""
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker=='primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        """
        :param kernel_type:
        :param dim:
        :param lamb:
        :param gamma:
        """
        self.kernel_type = kernel_type  #选用核函数的类型
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        """
        :param Xs: 源域的特征矩阵 （样本数x特征数）
        :param Xt: 目标域的特征矩阵 （样本数x特征数）
        :return: 经过TCA变换后的Xs_new,Xt_new
        """
        X = np.hstack((Xs.T, Xt.T))     #X.T是转置的意思；hstack则是将数据的相同维度数据放在一起
        X = X/np.linalg.norm(X, axis=0)  #求范数默认为l2范数即平方和开方，按列向量处理，这里相当于
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        #构造MMD矩阵 L
        e = np.vstack((1 / ns*np.ones((ns, 1)), -1 / nt*np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        #构造中心矩阵H
        H = np.eye(n) - 1 / n*np.ones((n, n))
        #构造核函数矩阵K
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n

        #注意核函数K就是后边的X特征，只不过用核函数的形式表示了
        a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye)#XMX_T+lamb*I
        b = np.linalg.multi_dot([K, H, K.T])#XHX_T

        w, V = scipy.linalg.eig(a, b)  #这里求解的是广义特征值，特征向量
        ind = np.argsort(w)#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到ind
        A = V[:, ind[:self.dim]]#取前dim个特征向量得到变换矩阵A，按照特征向量的大小排列好,
        Z = np.dot(A.T, K)#将数据特征*映射A
        Z /= np.linalg.norm(Z, axis=0)#单位向量话
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T#得到源域特征和目标域特征
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        """
        Transform Xs and Xt , then make p r edi c tion s on ta rg e t using 1NN
        param Xs : ns ∗ n_feature , source feature
        param Ys : ns ∗ 1 , source label
        param Xt : nt ∗ n_feature , target feature
        param Yt : nt ∗ 1 , target label
        return : Accuracy and predicted_labels on the target domain
        """
        Xs_new, Xt_new = self.fit(Xs, Xt)#经过TCA映射
        clf = KNeighborsClassifier(n_neighbors=1) #k近邻分类器，无监督学习
        clf.fit(Xs_new, Ys.ravel())#训练源域数据
        # 然后直接用于目标域的测试
        y_pred = clf.predict(Xs_new)
        acc = sklearn.metrics.accuracy_score(Ys, y_pred)
        print(acc)
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred

#%%
for labelsize in range(50,250,50):
    x1_train,y1_train=selectSample(x_train1,y_train1,labelsize)
    x2_train,y2_train=selectSample(x_train2,y_train2,labelsize)
    x3_train,y3_train=selectSample(x_train3,y_train3,labelsize)
    x4_train,y4_train=selectSample(x_train4,y_train4,labelsize)
    tca = TCA(kernel_type='rbf', dim=30, lamb=1, gamma=1)
    Xs_new, Xt_new =tca.fit(x2_train,x1_train)  
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(Xs_new, y2_train.ravel())#训练源域数据
    print("labelsize:",labelsize)
    y_pred = clf.predict(Xt_new)
    res=classification_report(y1_train,y_pred,target_names=['veg','shadow','water','road','building'],output_dict=True)
    print(np.array([res['veg']['f1-score'],res['shadow']['f1-score'],res['water']['f1-score'],res['road']['f1-score'],res['building']['f1-score'],res['weighted avg']['f1-score']]))
# %%
