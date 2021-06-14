import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions

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

true_labels=[]
pred_labels=[]              

#-----test_size=0.2
# random state =0,10,31,82 acc is 91.3
# random state =99 acc is 95.65
pca = PCA(2)


X_train, X_test, y_train, y_test = train_test_split(image_list,Label, test_size=0.2)
df = pca.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=3,weights='distance')#,algorithm='brute',leaf_size=15)
# With default parameters accuracy is 82.6
# with n_neighbors=2,weights='distance' acc is 86.95
knn.fit(df, y_train)

# score= knn.score( X_test,y_test)
# print(score*100)
df2 = pca.fit_transform(X_test)
plot_decision_regions(df2, y_test, clf=knn, legend=1)
pred_labels=knn.predict( df2)
print(accuracy_score( y_test,pred_labels)*100)
