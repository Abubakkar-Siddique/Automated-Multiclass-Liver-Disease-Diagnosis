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
from sklearn.svm import SVC

image_list = []
Label=[]

for image_path in glob.glob("./*.bmp"):   
    Label.append(image_path.split(' ')[2][0])

    image = mpimg.imread(image_path)
    size=image.size
    if (size!=4096):
        image=image[1:,1:]
        size=4096

    arr=np.reshape(image,size)
    image_list.append(arr)
    
image_array=np.array(image_list)/255

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



true_labels=[]
pred_labels=[]              


X_train, X_test, y_train, y_test = train_test_split(image_list,Label, test_size=0.25 )


svc=SVC()
svc.fit(X_train,y_train)
pred_labels=svc.predict(X_test)
print("SVC with rbf multi-class")
print(accuracy_score( y_test,pred_labels)*100)

svc.set_params(kernel='poly',degree=2)
svc.fit(X_train,y_train)
pred_labels=svc.predict(X_test)
print("SVC with poly multi-class")
print(accuracy_score( y_test,pred_labels)*100)

for f in range (len(image_list)):   #Encoding labels
    if(Label[f]==0):     #Normal
        pass
    else:
        Label[f]=1
X_train, X_test, y_train, y_test = train_test_split(image_list,Label, test_size=0.25 )

svc.set_params()
svc.fit(X_train,y_train)
pred_labels=svc.predict(X_test)
print("SVC with rbf binary-class")
print(accuracy_score( y_test,pred_labels)*100)

svc.set_params(kernel='poly',degree=2)
svc.fit(X_train,y_train)
pred_labels=svc.predict(X_test)
print("SVC with poly binary-class")
print(accuracy_score( y_test,pred_labels)*100)
















