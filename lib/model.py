print('importing keras...')
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
print('done.')

class ImageBranch(Sequential):

    def __init__(self, nchannels, aoisize):
        super().__init__()
        # inputSize = traindata.shape[-1]-1
        # train dataset passed in main
        img_x = aoisize[0]
        img_y = aoisize[1]

        input_shape = (img_x, img_y, nchannels)
        print('inputshape is ', input_shape)

        self.add(Conv2D(32, kernel_size=(5,5), strides=(1,1),
                        activation='relu',
                        input_shape=input_shape))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.add(Conv2D(64, (5,5), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2)))
        self.add(Flatten())
        #want to flatten it together with weather data in model class

        # self.add(Dense(32, activation='relu', input_dim=(input_shape)))
        # # self.add(Dropout(0.5))
        # self.add(Dense(1, activation='sigmoid'))

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        self.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    def fit(self, traindata, validatedata):

        traindata = traindata.astype('float32')
        validatedata = validatedata.astype('float32')

        # figure out normalization

        print('traindata shape', traindata.shape)
        print('validatedata shape', validatedata.shape)

        print('traindata:  ', traindata.shape[0])

        # guarantee the data is a 1D array of vectors
        # traindata = traindata.reshape(-1, traindata.shape[-1])
        # print('traindata reshape', traindata.shape)
        # the last entry in each vector is the output
        # inp = traindata[:,:-1]
        # out = traindata[:,-1]
        # super().fit(inp, out, epochs=100, batch_size=17500, validation_data=(inp, out))
        super().fit(inp, out, epochs=100, batch_size=17500, validation_data=(validatedata))

    def predict(self, data):
        # guarantee the data is a 1D array of vectors
        data = data.reshape(-1, data.shape[-1])
        # the last entry in each vector is the output
        inp = data[:,:-1]

        return super().predict(inp).flatten()
