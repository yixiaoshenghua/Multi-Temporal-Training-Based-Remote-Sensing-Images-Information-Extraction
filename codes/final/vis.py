#%%
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# %%
mapdir=r'G:\project\images\sentinal\process\output\map\17'
maplst=os.listdir(mapdir)
visdir=r'G:\project\images\sentinal\process\output\map\17visual'
#colors1=np.array([[127,255,0],[156,156,156],[255,245,152],[0,0,255],[255,0,0]])
#colors2=np.array([[127,255,0],[255,255,255],[255,245,152],[255,255,255],[255,255,255]])
colors3=np.array([[127,255,0],[255,255,255],[255,255,255],[255,255,255],[255,255,255]])
for m in tqdm(maplst):
    img=cv2.imread(os.path.join(mapdir, m),0)
    # vis1=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    # vis2=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    vis3=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    for i in range(5):
        #vis1[img==i]=colors1[i]
        #vis2[img==i]=colors2[i]
        vis3[img==i]=colors3[i]
    # cv2.imwrite(os.path.join(visdir,m),vis1.astype(np.uint8))
    # cv2.imwrite(os.path.join(visdir,'water_'+m),vis2.astype(np.uint8))
    cv2.imwrite(os.path.join(visdir,'veg_'+m),vis3.astype(np.uint8))
# %%
m=np.array([[0.2,0.4,0.6,1],[0,1,0.2,0.4],[0.8,0.4,0.6,0.2],[0.4,1,0.8,0.6]])
fig=plt.figure(figsize=(10,8))
#这就是所谓的第一种情况哦
h=plt.imshow(m,cmap='spring')
cb=plt.colorbar(h)
cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
font = {'family' : 'arial',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
        }
cb.set_label('confidence',fontdict=font) #设置colorbar的标签字体及其大小

# %%
