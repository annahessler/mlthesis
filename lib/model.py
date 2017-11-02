print('importing keras...')
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate, Input
from keras.optimizers import SGD, RMSprop
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
print('done.')


# class WeatherBranch(Sequential):

#     def __init__(self, inputSize):
#         super().__init__()
#         self.add(Dense(32, activation='relu', input_dim=(inputSize)))

class ImageBranch(Sequential):

    def __init__(self, nchannels, kernelSize):
        super().__init__()
        input_shape = (kernelSize, kernelSize, nchannels)

        self.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), input_shape=input_shape))
        self.add(Conv2D(32, kernel_size=(3,3), strides=(1,1),
                        activation='sigmoid'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.add(Dropout(0.5))
        self.add(Conv2D(64, (3,3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2)))
        self.add(Dropout(0.5))
        self.add(Flatten())
        self.add(Dense(128, activation='relu'))
        self.add(Dropout(0.5))

        self.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

class FireModel(Model):

    def __init__(self, InputSettings):
        self.InputSettings = InputSettings

        # print("creating network with shape", weatherDataSize, spatialChannels, aoiSize)
        self.wb = Input((self.variableSet.weatherDataSize,),name='weatherInput')
        self.ib = ImageBranch(spatialChannels, kernelSize)

        # print('weather branch info:', self.wb.shape)
        # print('image branch info:', self.ib.input_shape, self.ib.output_shape, self.ib.output)

        concat = Concatenate(name='mergedBranches')([self.wb,self.ib.output])
        out = Dense(1, kernel_initializer = 'normal', activation = 'sigmoid',name='output')(concat)
        # print("concat and out info:", concat.shape, out.shape)
        super().__init__([self.wb, self.ib.input], out)

        # self.add(Concatenate([self.wb, self.ib]))
        sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
        #rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])


    def fit(self, trainingSamples, validateSamples):
        inputs, outputs = trainingSamples.getData()
        history = super().fit(inputs, outputs, batch_size = 1000, epochs = 2, validation_data=validateData.getData())
        from time import localtime, strftime
        timeString = strftime("%d%b%H:%M", localtime())
        self.save('models/{}.h5'.format(timeString))
        return history

    def predict(self, dataset):
        inputs, outputs = dataset.getData()
        return super().predict(inputs).flatten()

class InputSettings(object):

    def __init__(self, AOIRadius=30, weatherMetrics=None, usedLayerNames=None):
        self.AOIRadius = AOIRadius
        self.weatherMetrics = weatherMetrics if weatherMetrics is not None else InputSettings.dummyMetric
        self.usedLayerNames = usedLayerNames if usedLayerNames is not None else 'all'

    @staticmethod
    def dummyMetric(weatherMatrix):
        return 42
