#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%
cor = np.array([[1,-0.94,0.96,-0.15,0.94,0.99,0.78,1,0.52,-0.67],
                [-0.94,1,-0.99,0.15,-1,-0.95,-0.53,-0.94,0.22,0.43],
                [0.96,-0.99,1,-0.15,0.99,0.95,0.61,0.96,0.32,-0.52],
                [-0.15,0.15,-0.15,1,-0.15,-0.15,-0.11,-0.15,-0.064,0.071],
                [0.94,-1,0.99,-0.15,1,0.95,0.53,0.94,0.22,-0.43],
                [0.99,-0.95,0.95,-0.15,0.95,1,0.75,0.99,0.48,-0.63],
                [0.70,-0.53,0.61,-0.11,0.53,0.75,1,0.78,0.92,-0.88],
                [1,-0.94,0.96,-0.15,0.94,0.99,0.75,1,0.52,-0.67],
                [0.52,-0.22,0.32,-0.064,0.22,0.48,0.92,0.52,1,-0.89],
                [-0.67,0.43,-0.52,0.071,-0.43,-0.63,-0.88,-0.67,-0.89,1]
],dtype=np.float32)
# %%
plt.figure(figsize=(11,8))
ytick_labels=['class','VARI','SAVI','MTVI','MSAVI','GNDVI','EVI','CIg','NDWI','NDVI']
xtick_labels=ytick_labels[::-1]
ytick_labels=ytick_labels[::-1]
xtick_location=list(range(10))
ytick_location=list(range(10))
plt.xticks(ticks=xtick_location, labels=xtick_labels,fontsize=15)
plt.yticks(ticks=ytick_location, labels=ytick_labels,fontsize=15)
plt.imshow(cor,cmap='OrRd_r')
plt.colorbar()
plt.tight_layout()
plt.savefig(r'G:\project\ISPRS\images\corr.jpg',dpi=200)


# %%
