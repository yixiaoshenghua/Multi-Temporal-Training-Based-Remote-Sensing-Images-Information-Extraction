#%%
from lib import *

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
# data1df['dis11']=np.zeros((data1df.shape[0]))
# data2df['dis21']=np.zeros((data2df.shape[0]))
# data1df['dis12']=np.zeros((data1df.shape[0]))
# data2df['dis22']=np.zeros((data2df.shape[0]))
# data1df['dis13']=np.zeros((data1df.shape[0]))
# data2df['dis23']=np.zeros((data2df.shape[0]))
# data1df['dis14']=np.zeros((data1df.shape[0]))
# data2df['dis24']=np.zeros((data2df.shape[0]))
datadf=pd.concat([data1df,data2df],axis=1)
# datadf['dis1Mesure']=np.zeros((datadf.shape[0]))
# datadf['dis2Mesure']=np.zeros((datadf.shape[0]))
# datadf['dis3Mesure']=np.zeros((datadf.shape[0]))
# datadf['dis4Mesure']=np.zeros((datadf.shape[0]))

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

# %%
# 构造model训练
coscores1=[]
coscores2=[]
sescores1=[]
sescores2=[]
for i in range(10):
    x1_train,y1_train=selectSample(x1,y1,5)
    x2_train,y2_train=selectSample(x2,y2,5)
    print("%d epoch"%i)
    model1=RandomForestClassifier(n_estimators=50)
    model2=RandomForestClassifier(n_estimators=50)
    x_train1=x1_train.copy()
    x_train2=x2_train.copy()
    y_train1=y1_train.copy()
    y_train2=y2_train.copy()
    model1,model2,coscore1,coscore2=run(datadf,x_train1,y_train1,x_train2,y_train2,model1,model2,x_test1,y_test1,x_test2,y_test2,size=10,epoch=20)
    # y_pred1=model1.predict(data1reshape).astype(np.int)
    # show1=color(y_pred1)
    # cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\cosample"+str((i+1)*1)+"epoch20size2class17.jpg",show1)
    # y_pred2=model2.predict(data2reshape).astype(np.int)
    # show2=color(y_pred2)
    # cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\cosample"+str((i+1)*1)+"epoch20size2class19.jpg",show2)
    coscores1.append(coscore1)
    coscores2.append(coscore2)
    print("%d epoch"%i)
    model3=RandomForestClassifier(n_estimators=50)
    model4=RandomForestClassifier(n_estimators=50)
    x_train3=x1_train.copy()
    x_train4=x2_train.copy()
    y_train3=y1_train.copy()
    y_train4=y2_train.copy()
    model3,model4,sescore1,sescore2=run(datadf,x_train3,y_train3,x_train4,y_train4,model3,model4,x_test1,y_test1,x_test2,y_test2,size=10,epoch=20,selflearn=True)
    # y_pred3=model3.predict(data1reshape).astype(np.int)
    # show3=color(y_pred3)
    # cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\selfsample"+str((i+1)*1)+"epoch20size2class17.jpg",show3)
    # y_pred4=model2.predict(data2reshape).astype(np.int)
    # show4=color(y_pred4)
    # cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\selfsample"+str((i+1)*1)+"epoch20size2class19.jpg",show4)
    sescores1.append(sescore1)
    sescores2.append(sescore2)

#%%
comax1=np.max(np.array(coscores1),axis=1)
comax2=np.max(np.array(coscores2),axis=1)
co1n=np.array(coscores1)[:,-1].ravel()
co2n=np.array(coscores1)[:,-1].ravel()
co10=np.array(coscores1)[:,0].ravel()
co20=np.array(coscores2)[:,0].ravel()
# %%
epochs=np.arange(1,21,dtype=np.int)
plt.figure(figsize=(10,8))
plt.plot(epochs,co10,color='black',marker='^',linestyle='-.')
plt.plot(epochs,co20,color='black',marker='o',linestyle='-.')
plt.plot(epochs,comax1,color='black',marker='^',linestyle='-')
plt.plot(epochs,comax2,color='black',marker='o',linestyle='-')

plt.ylim(0.8,1)
plt.xlim(0,20)
plt.xticks(epochs)
plt.ylabel('F1-score',fontsize=20)
plt.xlabel("label data size for each epoch",fontsize=20)
plt.legend(['T1 (T1&T2) before learning','T2 (T2&T2) before learning','T1 (T1&T2) after learning','T2 (T1&T2) after learning'],fontsize=16)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.savefig(r"E:\project\images\sentinal\process\output\coSize2Epoch20.jpg")
plt.show()

#%%
semax1=np.max(np.array(sescores1),axis=1)
semax2=np.max(np.array(sescores2),axis=1)
se10=np.array(sescores1)[:,0].ravel()
se20=np.array(sescores2)[:,0].ravel()
# %%
epochs=np.arange(1,21,dtype=np.int)
plt.figure(figsize=(10,8))
plt.plot(epochs,se10,color='black',marker='^',linestyle='-.')
plt.plot(epochs,se20,color='black',marker='o',linestyle='-.')
plt.plot(epochs,semax1,color='black',marker='^',linestyle='-')
plt.plot(epochs,semax2,color='black',marker='o',linestyle='-')
plt.ylim(0.8,1)
plt.xlim(0,20)
plt.xticks(epochs)
plt.ylabel('F1-score',fontsize=20)
plt.xlabel("label data size for each epoch",fontsize=20)
plt.legend(['T1 before learning','T2 before learning','T1 after learning','T2 after learning'],fontsize=16)
plt.tight_layout()
plt.savefig(r"E:\project\images\sentinal\process\output\seSize2Epoch20.jpg")
plt.show()

# %%
epochs=np.arange(1,21,dtype=np.int)
plt.figure(figsize=(10,8))
plt.plot(epochs,co1n-co10,color='black',marker='^',linestyle='-.')
plt.plot(epochs,co2n-co20,color='black',marker='o',linestyle='-.')
#plt.ylim(0.65,0.95)
plt.xlim(0,20)
plt.xticks(epochs)
plt.ylabel('F1-score',fontsize=20)
plt.xlabel("label data size for each epoch",fontsize=20)
plt.legend(['T1(T1&T2) PDC','T2(T1&T2) PDC'],fontsize=16)
plt.tight_layout()
plt.savefig(r"E:\project\images\sentinal\process\output\copdcSize2Epoch20.jpg")
plt.show()

# %%
