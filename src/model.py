from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import numpy
import os

class Model(Sequential):

    def __init__(self):
        self.add(Dense(32, activation='relu', input_dim=(11)))
        self.add(Dropout(0.5))
        self.add(Dense(1, activation='sigmoid'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        self.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    def fit(self, inp, out):
        super(self).fit(inp, out, epochs=500, batch_size=17500, validation_data=(inp, out))
