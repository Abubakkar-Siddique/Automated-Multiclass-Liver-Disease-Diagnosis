import glob

import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt
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


x_train, x_test, y_train, y_test = train_test_split(image_array, Label, test_size=0.25 )


model = Sequential()
model.add(Dense(48, input_dim=4096, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=7,epochs=15)
predictions = model.predict(x_test)





