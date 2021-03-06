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
            th1=c*class_candidate[i]['prob1'].quantile(q=0.5)
            th2=c*class_candidate[i]['prob2'].quantile(q=0.5)
            print(class_candidate[i]['prob1'].quantile(q=0.5),class_candidate[i]['prob2'].quantile(q=0.5))
            index1=np.array(class_candidate[i][class_candidate[i]['prob1']>th1].index)
            index2=np.array(class_candidate[i][class_candidate[i]['prob2']>th2].index)
            index=np.array([x for x in index1 if x in index2])
            print(index1,index2,index)
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
            th1=c*class_candidate1[i]['prob1'].quantile(q=0.5)
            th2=c*class_candidate2[i]['prob2'].quantile(q=0.5)
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
    if selflearn==False:
        for i in range(epoch):
            ii=curepoch*epoch+i
            model1.fit(x1,y1.ravel())
            model2.fit(x2,y2.ravel())
            y_pre1 = model1.predict(x_test1)
            y_pre2 = model2.predict(x_test2)
            pred1=model1.predict(datadf.loc[:,features1])
            pred2=model2.predict(datadf.loc[:,features2])
            y_pred1=model1.predict_proba(datadf.loc[:,features1])
            y_pred2=model2.predict_proba(datadf.loc[:,features2])
            y1_pred=model1.predict_proba(x1)
            y2_pred=model2.predict_proba(x2)
            resdf.at[ii,'coLoss1']=calcConsistencyLoss(datadf,y1.ravel(),y1_pred,y_pred1,pred2)
            resdf.at[ii,'coLoss2']=calcConsistencyLoss(datadf,y2.ravel(),y2_pred,y_pred2,pred1)
            resdf.at[ii,'coPDC']=calcPDC(datadf,pred1,pred2)
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
            model1.fit(x1,y1.ravel())
            model2.fit(x2,y2.ravel())
            y_pre1 = model1.predict(x_test1)
            y_pre2 = model2.predict(x_test2)
            pred1=model1.predict(datadf.loc[:,features1])
            pred2=model2.predict(datadf.loc[:,features2])
            y_pred1=model1.predict_proba(datadf.loc[:,features1])
            y_pred2=model2.predict_proba(datadf.loc[:,features2])
            y1_pred=model1.predict_proba(x1)
            y2_pred=model2.predict_proba(x2)
            resdf.at[ii,'seLoss1']=calcConsistencyLoss(datadf,y1.ravel(),y1_pred,y_pred1,pred2)
            resdf.at[ii,'seLoss2']=calcConsistencyLoss(datadf,y2.ravel(),y2_pred,y_pred2,pred1)
            resdf.at[ii,'sePDC']=calcPDC(datadf,pred1,pred2)
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
        model1=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=2
                                  )
        model2=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=2
                                  )
        x_train1=x1_train.copy()
        x_train2=x2_train.copy()
        y_train1=y1_train.copy()
        y_train2=y2_train.copy()
        model1,model2,resdf=run(resdf,datadf,x_train1,y_train1,x_train2,y_train2,model1,model2,x_test1,y_test1,x_test2,y_test2,c=1.0,size=size,epoch=epoch,selflearn=False,curepoch=i)
        # y_pred1=model1.predict(data1reshape).astype(np.int)
        # show1=color(y_pred1)
        # cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\cosample"+str((i+1)*1)+"epoch20size2class17.jpg",show1)
        # y_pred2=model2.predict(data2reshape).astype(np.int)
        # show2=color(y_pred2)
        # cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\cosample"+str((i+1)*1)+"epoch20size2class19.jpg",show2)
        print("%d epoch"%i)
        # model3=RandomForestClassifier(n_estimators=50)
        # model4=RandomForestClassifier(n_estimators=50)
        model3=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=2
                                  )
        model4=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=2
                                  )
        x_train3=x1_train.copy()
        x_train4=x2_train.copy()
        y_train3=y1_train.copy()
        y_train4=y2_train.copy()
        model3,model4,resdf=run(resdf,datadf,x_train3,y_train3,x_train4,y_train4,model3,model4,x_test1,y_test1,x_test2,y_test2,c=1.0,size=size,epoch=epoch,selflearn=True,curepoch=i)
        # y_pred3=model3.predict(data1reshape).astype(np.int)
        # show3=color(y_pred3)
        # cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\selfsample"+str((i+1)*1)+"epoch20size2class17.jpg",show3)
        # y_pred4=model2.predict(data2reshape).astype(np.int)
        # show4=color(y_pred4)
        # cv2.imwrite(r"E:\project\images\sentinal\process\output\cotrain\selfs
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

def calcConsistencyLoss(datadf,y_true,y_pred,y_pred1,y2_pred,lamda=1.0):
    labeled_loss=log_loss(y_true,y_pred,labels=[0,1,2])
    unlabeled_loss=log_loss(y2_pred,y_pred1,labels=[0,1,2])
    loss=labeled_loss+unlabeled_loss*lamda
    return loss
