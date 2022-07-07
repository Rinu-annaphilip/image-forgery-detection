import pickle
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import random
import json

datalist=pickle.load(open("datalist1.pkl","rb"))
label=pickle.load(open("labels1.pkl","rb"))
print(label)
print(len(label))

X = np.array(datalist)
Y = np.array(label)
X = X.reshape(-1, 128, 128, 3) 



X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)
X = X.reshape(-1,1,1,1)




def CNN_model():
    model = Sequential()  # Sequential Model
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), padding='same'))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), padding='same'))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), padding='same'))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), padding='same'))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), padding='same'))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation = 'sigmoid'))
    return model

model = CNN_model()
model.summary()

epochs = 20
batch_size = 32
#Optimizer
init_lr = 1e-4   #learning rate for the optimizer
optimizer = Adam(lr = init_lr, decay = init_lr/epochs) 
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])


#Early Stopping
early_stopping = EarlyStopping(monitor = 'val_accuracy',
                               min_delta = 0,
                               patience = 10,
                               verbose = 0,
                               mode = 'auto')
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50)
# hist = model.fit(X_train,
#                  Y_train,
#                  batch_size = batch_size,
#                  epochs = epochs,
#                  sameation_data = (X_val, Y_val),
#                  callbacks = [early_stopping])

#save the model as a h5 file
model.save('cnnmodel.h5') 