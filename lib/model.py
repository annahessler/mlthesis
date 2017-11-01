print('importing keras...')
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate, Input
from keras.optimizers import SGD, RMSprop
from keras.layers import Conv2D, MaxPooling2D
from lib import histories
from lib import viz
print('done.')


# class WeatherBranch(Sequential):

#     def __init__(self, inputSize):
#         super().__init__()
#         self.add(Dense(32, activation='relu', input_dim=(inputSize)))

class ImageBranch(Sequential):

    def __init__(self, nchannels, kernelSize):
        super().__init__()
        input_shape = (kernelSize, kernelSize, nchannels)

        self.add(Conv2D(32, kernel_size=(3,3), strides=(1,1),
                        activation='sigmoid',
                        input_shape=input_shape))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.add(Conv2D(64, (3,3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2)))
        self.add(Flatten())
        self.add(Dense(128, activation='relu'))

        self.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])


class FireModel(Model):

    def __init__(self, weatherDataSize, spatialChannels, kernelSize):
        # print("creating network with shape", weatherDataSize, spatialChannels, aoiSize)
        self.wb = Input((weatherDataSize,),name='weatherInput')
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
        self.history = None


    def fit(self, trainData, validateData, testData):

        self.history = histories.Histories(validateData)
        print("history callback losses are " , self.history.losses)
        inputs, outputs = trainData.getData()
        super().fit(inputs, outputs, batch_size = 1000, epochs = 5, validation_data=validateData.getData(), callbacks=[self.history])
        # print("after history callback losses are " , self.history.__dir__())
        # viz.visualize_training(self.history, 'testviz', validateData.getData(), trainData)
        from time import localtime, strftime
        timeString = strftime("%d%b%H:%M", localtime())
        self.save('models/{}.h5'.format(timeString))
        

    def predict(self, dataset):
        inputs, outputs = dataset.getData()
        return super().predict(inputs).flatten()
