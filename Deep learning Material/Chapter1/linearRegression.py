# Author: Vivek Singh

# Simple linear regression model

import numpy as np

x1=np.random.rand(100).astype(np.float32)
x2=np.random.rand(100).astype(np.float32)

x_train=np.column_stack((x1,x2))
y=x1*0.1+x2*0.3 +0.5

from keras.models import Sequential

model=Sequential()

from keras.layers import Dense, Activation

model.add(Dense(units=10, input_dim=2))
model.add(Activation('relu'))

model.add(Dense(units=1))
model.add(Activation('softmax'))


model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train,y)
