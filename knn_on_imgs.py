
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

image_list = []
Label=[]
# img_index=0
# names= []

for image_path in glob.glob("./images/*.bmp"):   
    Label.append(image_path.split(' ')[2][0])
#     names.append(image_path[-1])
    image = mpimg.imread(image_path)
    size=image.size
    if (size!=4096):
        image=image[1:,1:]
        size=4096
    # arr=np.array(image)
    arr=np.reshape(image,size)
    image_list.append(arr)
    
# images=len(image_list)
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
loo = LeaveOneOut()
test_arr=[]

for train_index, test_index in loo.split(image_list):
# split into train and test
    # print("train_index ",train_index,"test_index ",test_index)
    X_train, y_train = image_array[train_index], Label[train_index]
    X_test, y_test = image_array[test_index], Label[test_index]
    knn = KNeighborsClassifier(n_neighbors=1,weights='distance', p=1)       # n_neighbors=1 acc is 85.96
     # n_neighbors=2,weights='distance' acc is 85.96                   # n_neighbors=2 acc is 84.80
     # n_neighbors=3,weights='distance' acc is 81.87                                              # n_neighbors=3 &4 acc is 81.29
     # n_neighbors>3,weights='distance' acc decreases                                               # n_neighbors=5 acc decreases to 79.53
     # algorithm='ball_tree/ kd_tree', leaf_size= ? acc remains 85.96                                                                    # n_neighbors>5 acc decreases 
    test_arr.append(np.array(X_test))
    # fitting the model
    knn.fit(X_train, y_train)
    # predict the response
    pred= knn.predict(X_test)
    pred_labels.append(pred[0])
    true_labels.append(y_test[0])
    
    if (y_test!=pred):                  
        mis_class+=1                    
    if(y_test == 2 and pred==2):        #
        fattp+=1                        #   Check for correctly classified instances  (true positives)for all three cases
    elif(y_test == 1 and pred==1):      #
        hettp+=1                        #
    elif(y_test == 0 and pred==0):      #
        nortp+=1                        #
        
    #--------------------------------------------------------
    if(y_test == 2 and pred!=2):
        fatfn+=1
    elif(y_test == 1 and pred!=1):  #check for false naegatives for all three cases
        hetfn+=1
    elif(y_test == 0 and pred!=0):
        norfn+=1
    #-------------------------------------------------------------        
    if(y_test != 2 and pred!=2):
        fattn+=1
    if(y_test != 1 and pred!=1): #check for true negatives
        hettn+=1
    if(y_test != 0 and pred!=0):
        nortn+=1
        
    #-----------------------------------------------------------      
    if(y_test != 2 and pred==2):
        fatfp+=1                    #check for false positives
        if(y_test==1):fat_het_fp+=1
        if(y_test==0):fat_nor_fp+=1

    if(y_test != 1 and pred==1):
        hetfp+=1
        if(y_test==2):het_fat_fp+=1
        if(y_test==0):het_nor_fp+=1
    if(y_test != 0 and pred==0):
        norfp+=1
        if(y_test==2):nor_fat_fp+=1
        if(y_test==1):nor_het_fp+=1
        
fpred_count=pred_labels.count(2)
hpred_count=pred_labels.count(1)
npred_count=pred_labels.count(0)

print ("number of predicted Fatty ,Hetrogenous & Normal labels are ", fpred_count,hpred_count,npred_count)
print ("number of true positive predicted Fatty ,Hetrogenous & Normal labels are ", fattp, hettp,nortp)
print ("number of false positive predicted Fatty ,Hetrogenous & Normal labels are ", fatfp, hetfp,norfp)
print ("number of true negative predicted Fatty ,Hetrogenous & Normal labels are ", fattn, hettn,nortn)
print ("number of false negative predicted Fatty ,Hetrogenous & Normal labels are ", fatfn, hetfn,norfn)

cmat= confusion_matrix(true_labels, pred_labels)
classes=["normal", "hetro","fatty" ]
thresh = cmat.max() / 2.

for i, j in itertools.product(range(cmat.shape[0]), range(cmat.shape[1])):
    
    plt.text(j, i, cmat[i, j], horizontalalignment="center")#,color="white" if cmat[i, j] > thresh else "black")
tick_marks = np.arange(len(classes))
# plt.tight_layout()
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.imshow(cmat, cmap=plt.cm.rainbow) 

print ("number of false positive for fatty as hetro & normal are ", fat_het_fp, fat_nor_fp)
print ("number of false positive for hetro as fatty & normal are ", het_fat_fp, het_nor_fp)
print ("number of false positive for normal as fatty & hetro are ", nor_fat_fp, nor_het_fp)


all_TP=fattp+ hettp+nortp
all_TN=fattn+hettn+nortn
all_FP=fatfp+ hetfp+norfp
all_FN=fatfn+hetfn+norfn

print ("number of mis_classification :", mis_class)        

sen_fat= 100*(float(fattp)/float(fattp+fatfn))
print ("sensitivity of fatty is ==%.2f" % sen_fat , "%")

sen_het= 100*(float(hettp)/float(hettp+hetfn))
print ("sensitivity of hetro is ==%.2f" % sen_het , "%")

sen_nor= 100*(float(nortp)/float(nortp+norfn))
print ("sensitivity of fatty, hetro, normal is ==%.2f" % sen_fat , "%", " %.2f" % sen_het ,"%",  " %.2f" %sen_nor, "%")


spe_fat= 100*(float(fattn)/float(fattn+fatfp))
print ("specificity of fatty is ==%.2f" % spe_fat , "%")

spe_het= 100*(float(hettn)/float(hettn+hetfp))
print ("specificity of hetro is ==%.2f" % spe_het , "%")

spe_nor= 100*(float(nortn)/float(nortn+norfp))
print ("specificity of fatty, hetro, normal is ==%.2f" % spe_fat , "%", " %.2f" % spe_het ,"%",  " %.2f" % spe_nor , "%")

pre_fat= 100*(float(fattp)/float(fattp+fatfp))
print ("precision of fatty is ==%.2f" %pre_fat , "%")

pre_het= 100*(float(hettp)/float(hettp+hetfp))
print ("precision of hetro is ==%.2f" % pre_het , "%")

pre_nor= 100*(float(nortp)/float(nortp+norfp))
print ("precision of fatty, hetro, normal is ==%.2f" % pre_fat , "%", " %.2f" % pre_het ,"%",  " %.2f" % pre_nor , "%")

Acc=accuracy_score(Label,pred_labels)*100
print("\nAccuracy Score for Knn using loo:")
print("===============\n")
print(Acc)

