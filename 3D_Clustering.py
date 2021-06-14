# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:44:24 2021

@author: Abubakkar Siddique
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:26:51 2021

@author: Abubakkar Siddique
"""

import glob
import numpy as np
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import plotly.express as px
from plotly.offline import plot

# plt.style.use('default')
# color_pallete = ['r', 'g', 'b']
# sns.set_palette(color_pallete)
# sns.set_style("white")



image_list = []
Label=[]

for image_path in glob.glob("./images/*.bmp"):   
    Label.append(image_path.split(' ')[2][0])

    image = mpimg.imread(image_path)
    size=image.size
    if (size!=4096):
        image=image[1:,1:]
        size=4096

    arr=np.reshape(image,size)
    image_list.append(arr)
    
image_array=np.array(image_list)

for f in range (len(image_list)):   #Encoding labels
    if(Label[f]=='F'):
        Label[f]=2
    elif (Label[f] =='N'):
        Label[f]=0
    else:
        Label[f]=1
f_count=Label.count(2)
h_count=Label.count(1)
n_count=Label.count(0)

Label=np.array(Label)

df = pd.DataFrame(Label, columns = ['Label'])



pca = PCA(n_components=6)
principalComponents = pca.fit_transform(image_array)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5','principal component 6'])
finalDf = pd.concat([principalDf, df], axis = 1)


plt.figure(figsize=(8, 8))

ax = sns.pairplot(finalDf, hue='Label')

plt.show()

fig = px.scatter_3d(finalDf, x='principal component 1', y='principal component 3', z='principal component 5',  color="Label", symbol='Label')

plot(fig)







