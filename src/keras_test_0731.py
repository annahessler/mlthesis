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


m = Model()
# Train the model, iterating on the data in batches of 32 samples
m.fit(train_today, train_tomorrow)

predictions_test = m.predict(test_today)
print(predictions_test)

predictions_train = m.predict(train_today)
print(predictions_train)

numpy.savetxt('output/predictions_test.csv', predictions_test, delimiter = ',')
numpy.savetxt('output/predictions_train.csv', predictions_train, delimiter = ',')


# evaluate the model
scores = m.evaluate(test_today, test_tomorrow)
print("\n%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
