import os
import cv2
def genlabelimg(path,a1,a2,a3,a4):
    dir,name=os.path.split(path)
    data=np.loadtxt(open(path,'r'),delimiter=',',skiprows=21,dtype=np.uint8)
    loc=data[:,:2]
    label=np.zeros((256,256),dtype=np.uint8)
    for i in range(loc.shape[0]):
        if i<a1:
            label[loc[i,1],loc[i,0]]=1
        elif i<a1+a2:
            label[loc[i,1],loc[i,0]]=2
        elif i<a1+a2+a3:
            label[loc[i,1],loc[i,0]]=3
        elif i<a1+a2+a3+a4:
            label[loc[i,1],loc[i,0]]=4
    cv2.imwrite(dir+"\\"+name.split('.')[0]+'.tif',label)
    cv2.imwrite(dir+"\\visual\\"+name.split('.')[0]+'.tif',label*50)
    return True