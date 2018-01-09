
# coding: utf-8

# In[2]:

from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671)


# In[4]:

NB_EPOCH=200
BATCH_SIZE=128
VERBOSE=1
NB_CLASSES=10
OPTIMIZER=SGD()
N_HIDDEN=128
VALIDATION_SPLIT=0.2

(X_train,y_train),(X_test,y_test)=mnist.load_data()


# In[7]:

RESHAPED=784
X_train=X_train.reshape(60000, RESHAPED)
X_test=X_test.reshape(10000, RESHAPED)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=255
X_test/=255
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
Y_train=np_utils.to_categorical(y_train,NB_CLASSES)
Y_test= np_utils.to_categorical(y_test,NB_CLASSES)


# In[8]:

model=Sequential()
model.add(Dense(NB_CLASSES,input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()


# In[9]:

model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])


# In[10]:

history=model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)


# In[12]:

score=model.evaluate(X_test,Y_test,verbose=VERBOSE)
print("Test Score:",score[0])
print('Test accuracy',score[1])


# In[ ]:



