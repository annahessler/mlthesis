from time import localtime, strftime

print('importing keras...')
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate, Input
from keras.optimizers import SGD, RMSprop
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
print('done.')

from lib import preprocess
from lib import metrics

class ImageBranch(Sequential):
    print("model.py")

    def __init__(self, nchannels, kernelDiam):
        super().__init__()
        # there is also the starting perim which is implicitly gonna be included
        nchannels += 1
        input_shape = (kernelDiam, kernelDiam, nchannels)

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

    def __init__(self, preProcessor, weightsFileName=None):
        self.preProcessor = preProcessor

        kernelDiam = 2*self.preProcessor.AOIRadius+1
        self.wb = Input((self.preProcessor.numWeatherInputs,),name='weatherInput')
        self.ib = ImageBranch(len(self.preProcessor.whichLayers), kernelDiam)

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

        if weightsFileName is not None:
            self.load_weights(weightsFileName)


    def fit(self, training, validate, epochs=1):
        # get the actual samples from the collection of points
        (tinputs, toutputs), ptList = self.preProcessor.process(training)
        (vinputs, voutputs), ptList = self.preProcessor.process(validate)
        print('training on ', training)
        history = super().fit(tinputs, toutputs, batch_size = 1000, epochs=80, validation_data=(vinputs, voutputs))

        self.saveWeights()
        return history

    def saveWeights(self, fname=None):
        if fname is None:
            timeString = strftime("%d%b%H:%M", localtime())
            fname = 'models/{}.h5'.format(timeString)
        self.save_weights(fname)

    def predict(self, dataset):
        (inputs, outputs), ptList = self.preProcessor.process(dataset)
        results = super().predict(inputs).flatten()
        resultDict = {pt:pred for (pt, pred) in zip(ptList, results)}
        return resultDict

from collections import namedtuple
InputSettings = namedtuple('InputSettings', ['usedLayerNames', 'weatherMetrics', 'AOIRadius'])

# class InputSettings(object):
#     '''This things which define the inputs to a Model:
#     -weatherMetrics: How many and which weather metrics we use
#     -usedLayerNames: Which layers get fed in (dem, nir, etc.)
#     -AOIRadius: How large of a window around each sample do we feed into the CNN'''
#
#     def __init__(self, usedLayerNames, weatherMetric, AOIRadius=30):
#         self.usedLayerNames = usedLayerNames
#         self.weatherMetrics = weatherMetric
#         self.AOIRadius = AOIRadius
#         assert type(self.usedLayerNames) == list
#         assert len(usedLayerNames) > 0
