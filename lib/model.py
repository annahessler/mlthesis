print('importing keras...')
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Merge
from keras.optimizers import SGD
print('done.')

class WeatherBranch(Sequential):

    def __init__(self, inputSize):
        super().__init__()
        self.add(Dense(32, activation='relu', input_dim=(inputSize)))
        self.add(Dropout(0.5))
        self.add(Dense(1, activation='sigmoid'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        self.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

class Model(Sequential):

    def __init__(self, weatherDataSize, spatialChannels, aoiSize):

        self.wb = WeatherBranch(weatherDataSize)
        self.ib = ImageBranch(spatialChannels, aoiSize)

        self.add(Merge([self.wb, self.ib], mode = 'concat'))
        self.add(Dense(1, init = 'normal', activation = 'sigmoid'))
        sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
        self.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])


