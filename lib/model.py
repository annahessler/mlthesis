print('importing keras...')
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
print('done.')

class Model(Sequential):

    def __init__(self, data):
        super().__init__()
        inputSize = data.shape[-1]-1
        self.add(Dense(32, activation='relu', input_dim=(inputSize)))
        self.add(Dropout(0.5))
        self.add(Dense(1, activation='sigmoid'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        self.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    def fit(self, data):
        # guarantee the data is a 1D array of vectors
        data = data.reshape(-1, data.shape[-1])
        # the last entry in each vector is the output
        inp = data[:,:-1]
        out = data[:,-1]
        super().fit(inp, out, epochs=100, batch_size=17500, validation_data=(inp, out))

    def predict(self, data):
        # guarantee the data is a 1D array of vectors
        data = data.reshape(-1, data.shape[-1])
        # the last entry in each vector is the output
        inp = data[:,:-1]

        return super().predict(inp).flatten()