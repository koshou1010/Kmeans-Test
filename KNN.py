#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import read_excel
import numpy as np
from numpy import arange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode


# In[2]:


dataset = read_excel('YuData9.xlsx')


# In[3]:


#print(dataset.tail())
dataset.fillna(0)
cols=dataset.columns[1:]


# In[4]:


#print(cols)
np.isnan(dataset).any()
dataset = dataset.fillna(method='ffill')


# In[5]:


#設定分幾群
clusters=KMeans(n_clusters=3)
dataset["clusters"]=clusters.fit_predict(dataset[cols])
#顯示前10筆
dataset.head(10)


# In[6]:


#Principal component separation to create 2 dim picture
pca=PCA(n_components=2)
dataset['x']=pca.fit_transform(dataset[cols])[:,0]
dataset['y']=pca.fit_transform(dataset[cols])[:,1]
dataset=dataset.reset_index()
print(dataset.tail())


# In[7]:


#設定各個點的大小以及分3群顏色

trace0= go.Scatter(x=dataset[dataset.clusters == 0]['x'],
                   y=dataset[dataset.clusters == 0]['y'],
                   name="Cluster1",
                   mode ="markers",
                   marker =dict(size=10,color="rgba(15,152,152,0.5)",line=dict(width=1,color="rgb(0,0,0)")))
trace1= go.Scatter(x=dataset[dataset.clusters == 1]['x'],
                   y=dataset[dataset.clusters == 1]['y'],
                   name="Cluster2",
                   mode ="markers",
                   marker =dict(size=10,color="rgba(180,18,180,0.5)",line=dict(width=1,color="rgb(0,0,0)")))
trace2= go.Scatter(x=dataset[dataset.clusters == 2]['x'],
                   y=dataset[dataset.clusters == 2]['y'],
                   name="Cluster3",
                   mode ="markers",
                   marker =dict(size=10,color="rgba(132,132,132,0.8)",line=dict(width=1,color="rgb(0,0,0)")))


# In[8]:


data =[trace0,trace1,trace2]

iplot(data)


# In[13]:


import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

img = image.load_img("C:/Users/123/newplot.PNG")
plt.figure(figsize=(20, 10))
plt.imshow(img)


# In[ ]:




