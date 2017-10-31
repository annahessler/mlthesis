print('importing keras...')
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate, Input
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
print('done.')


# class WeatherBranch(Sequential):

#     def __init__(self, inputSize):
#         super().__init__()
#         self.add(Dense(32, activation='relu', input_dim=(inputSize)))

class ImageBranch(Sequential):

    def __init__(self, nchannels, aoisize):
        super().__init__()
        img_x = aoisize[0]
        img_y = aoisize[1]
        input_shape = (img_x, img_y, nchannels)

        self.add(Conv2D(32, kernel_size=(3,3), strides=(1,1),
                        activation='relu',
                        input_shape=input_shape))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # self.add(Conv2D(64, (5,5), activation='relu'))
        # self.add(MaxPooling2D(pool_size=(2,2)))
        self.add(Flatten())
        self.add(Dense(128, activation='relu'))

        self.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

class FireModel(Model):

    def __init__(self, weatherDataSize, spatialChannels, aoiSize):
        print("creating network with shape", weatherDataSize, spatialChannels, aoiSize)
        self.wb = Input((weatherDataSize,))
        self.ib = ImageBranch(spatialChannels, aoiSize)

        print('weather branch info:', self.wb.shape)
        print('image branch info:', self.ib.input_shape, self.ib.output_shape, self.ib.output)

        concat = Concatenate()([self.wb,self.ib.output])
        out = Dense(1, kernel_initializer = 'normal', activation = 'sigmoid')(concat)
        print("concat and out info:", concat.shape, out.shape)
        super().__init__([self.wb, self.ib.input], out)

        # self.add(Concatenate([self.wb, self.ib]))
        sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
        self.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])


    def fit(self, trainData):
        weather,spatialData, outputs = trainData.getData()
        super().fit([weather, spatialData], outputs, batch_size = 2000, epochs = 50, verbose = 1)

    def predict(self, dataset):
        weather,spatialData, outputs = dataset.getData()
        return super().predict([weather, spatialData])
