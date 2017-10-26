from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import numpy
import os

os.chdir('..')
dataset = numpy.loadtxt('data/forModel/0731.csv', delimiter = ',')
today = dataset[:,0:11]
tomorrow = dataset[:,11]

splitSize = int(dataset.shape[0]*.6)
trainSet = dataset[:splitSize]
testSet = dataset[splitSize:]
print(trainSet, testSet, trainSet.shape, testSet.shape)

train_today, train_tomorrow = trainSet[:,0:11], trainSet[:,11]
test_today, test_tomorrow = testSet[:,0:11], testSet[:,11]

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=(11)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
model.fit(train_today, train_tomorrow, epochs=500, batch_size=17500, validation_data=(test_today, test_tomorrow))

predictions_test = model.predict(test_today)
print(predictions_test)

predictions_train = model.predict(train_today)
print(predictions_train)

numpy.savetxt('output/predictions_test.csv', predictions_test, delimiter = ',')
numpy.savetxt('output/predictions_train.csv', predictions_train, delimiter = ',')


# evaluate the model
scores = model.evaluate(test_today, test_tomorrow)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
