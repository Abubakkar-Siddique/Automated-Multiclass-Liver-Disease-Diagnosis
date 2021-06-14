import cv2
import glob
import random
import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score

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

print ("########### Nearest Neighbour classifier #######")

print ("Total number of Fatty ,Hetro & Normal labels are ", f_count,h_count,n_count)

mis_class=0
fattp=0
fattn=0
fatfp=0
fatfn=0
fat_het_fp=0
fat_nor_fp=0


hettp=0
hettn=0
hetfp=0
hetfn=0
het_fat_fp=0
het_nor_fp=0

norfp=0
nortp=0
nortn=0
norfn=0
nor_fat_fp=0
nor_het_fp=0

true_labels=[]
pred_labels=[]              

#-----test_size=0.2
# random state =0,10,31,82 acc is 91.3
# random state =99 acc is 95.65

X_train, X_test, y_train, y_test = train_test_split(image_list,Label, test_size=0.2, )
knn = KNeighborsClassifier(n_neighbors=1,weights='distance')#,algorithm='brute',leaf_size=15)
# With default parameters accuracy is 82.6
# with n_neighbors=2,weights='distance' acc is 86.95
knn.fit(X_train, y_train)
# pred_labels=knn.predict(X_test)
score= knn.score( X_test,y_test)
print("Accuracy of knn using train test")
print(score*100)
# print(accuracy_score( y_test,pred_labels)*100)
