from keras.models import Sequential

model=Sequential()

from keras.layers import Dense, Activation

# add dense layer. In dense layer, each neuron is connected to all the neurons in next layer
model.add(Dense(units=64, input_dim=5))

# add activation function for each layer. Each layer has its activation function. It represents the function 
# applied on the output of the neuron
model.add(Activation('relu'))

# Similarly, lets add one more layer with 10 neuron
model.add(Dense(units=10))

# add a activation function
model.add(Activation('softmax'))


# Similarly, lets add one more layer with 1 neuron
model.add(Dense(units=1))

# add a activation function
model.add(Activation('softmax'))


# compile the above model of 2 layers
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']) 

# Now we need to create a numpy matrix x_train and a numpy array y_train

import numpy as np

x_train=np.random.rand(100,5)
y_train=np.random.rand(100,)

model.fit(x_train, y_train, epochs=5, batch_size=32)

