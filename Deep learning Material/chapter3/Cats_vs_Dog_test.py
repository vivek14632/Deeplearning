# Authors:

# Data source: https://www.kaggle.com/c/dogs-vs-cats/data

# Vivek: What is the use of ImageDataGenerator?
from keras.preprocessing.image import ImageDataGenerator

# Vivek: Are there models other than sequential?
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

# Vivek: What is the purpose of backend?
from keras import backend as K

# dimensions of our images.
img_width, img_height = 150, 150

# Vivek: Any public link for these images so that others can download?
train_data_dir = 'C:/Users/preks/Documents/data/train'

validation_data_dir = 'C:/Users/preks/Documents/data/validation'

nb_train_samples = 2000

nb_validation_samples = 800

# Vivek: what is the importance of epochs and batch_size?
epochs = 25
batch_size = 16


# Vivek: Not clear what exactly are we doing here?
# Vivek: What is 'channels_first'? It seems like whether the color is x dimension 
# or z dimension?
if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

model = Sequential()

# Convolution layer with 32 units.
model.add(Conv2D(32, (3, 3), input_shape=input_shape))

# RELU activation
model.add(Activation('relu'))

# Max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(32, (3, 3)))

# 2nd activation layer
model.add(Activation('relu'))

# 2nd max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolution layer. 
# Vivek: why did we increase the units to 64? The earlier convolution layers 
# have 32 units
model.add(Conv2D(64, (3, 3)))

# Third activation layer
model.add(Activation('relu'))

# Third pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Vivek: What is the purpose of flatten layer?
model.add(Flatten())

# Dense layer
model.add(Dense(64))

# Final relu activation
model.add(Activation('relu'))

# Dropout layer to reduce overfitting
model.add(Dropout(0.5))

# Output layer, since its just 2 classes so one neuron is enought (0/1)
model.add(Dense(1))

# sigmoid activation similar to a logistic regression
model.add(Activation('sigmoid'))

# Compile the above model 
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2,
    rotation_range=20,# angle at which the image is rotated
width_shift_range=0.5, # randomly translate as a fraction of total width or height
height_shift_range=0.5,

    horizontal_flip=True)



# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')



model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=nb_validation_samples // batch_size)


model.save_weights('first_try.h5')


# In[165]:

import os
cwd=os.getcwd()
cwd


# In[166]:

import h5py    
import numpy as np    
f1 = h5py.File('C:/Users/preks/first_try.h5','r+')


# In[167]:

f1


# In[168]:

f1.keys()


# In[169]:

from keras.models import load_model


# In[170]:

from keras.preprocessing.image import img_to_array, load_img


# In[171]:

model.save('C:/Users/preks/Desktop/first_try.h5')
test_model = load_model('first_try.h5')


# In[172]:

img = load_img('image_predict.jpg',False,target_size=(img_width,img_height))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)


# In[173]:

preds = test_model.predict_classes(x)


# In[174]:

prob = test_model.predict_proba(x)
print(preds, prob)
