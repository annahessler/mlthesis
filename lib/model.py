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

class ImageBranch(Sequential):

    def __init__(self, nchannels, aoisize):
        super().__init__()
        # inputSize = traindata.shape[-1]-1
        # train dataset passed in main
        img_x = aoisize[0]
        img_y = aoisize[1]

        input_shape = (img_x, img_y, nchannels)

        self.add(Conv2D(128, kernel_size=(5,5), strides=(1,1),
                        activation='relu',
                        input_shape=input_shape))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.add(Conv2D(64, (5,5), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2)))

        self.add(Flatten())
        #want to flatten it together with weather data in model class


        # self.add(Dense(32, activation='relu', input_dim=(inputSize)))
        # self.add(Dropout(0.5))
        # self.add(Dense(1, activation='sigmoid'))

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        # self.compile(optimizer='rmsprop',
        #       loss='binary_crossentropy',
        #       metrics=['accuracy'])



class Model(Sequential):

    def __init__(self, weatherDataSize, spatialChannels, aoiSize):

        self.wb = WeatherBranch(weatherDataSize)
        self.ib = ImageBranch(spatialChannels, aoiSize)

        self.add(Merge([self.wb, self.ib], mode = 'concat'))
        self.add(Dense(32, init = 'normal', activation = 'sigmoid'))
        self.add(Dense(1, init = 'normal', activation = 'relu'))
        sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
        self.compile(loss = 'binary_crossentropy', optimizer = Adam(), metrics = ['accuracy'])


    def fit(self, weatherData, spatialData, outputs):

        super().fit([weather, spatialData], outputs, batch_size = 2000, nb_epoch = 100, verbose = 1)

        # traindata = traindata.astype('float32')
        # testdata = testdata.astype('float32')

        # # figure out normalization

        # print('traindata shape', traindata.shape)
        # print('testdata shape', testdata.shape)
        # # guarantee the data is a 1D array of vectors
        # traindata = traindata.reshape(-1, data.shape[-1])
        # print('traindata reshape', traindata.shape)
        # # the last entry in each vector is the output
        # inp = data[:,:-1]
        # out = data[:,-1]
        # super().fit(inp, out, epochs=100, batch_size=17500, validation_data=(inp, out))

    def predict(self, data):
        # guarantee the data is a 1D array of vectors
        data = data.reshape(-1, data.shape[-1])
        # the last entry in each vector is the output
        inp = data[:,:-1]

        return super().predict(inp).flatten()
