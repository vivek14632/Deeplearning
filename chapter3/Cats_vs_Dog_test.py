
# coding: utf-8

# In[4]:

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K





# dimensions of our images.

img_width, img_height = 150, 150



train_data_dir = 'C:/Users/preks/Documents/data/train'

validation_data_dir = 'C:/Users/preks/Documents/data/validation'

nb_train_samples = 2000

nb_validation_samples = 800

epochs = 25

batch_size = 16



if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,
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


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



