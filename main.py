
import numpy as np
<<<<<<< HEAD
# import lib.model
from lib.datamodule import Data
from lib.dataset import Dataset

data = Data.defaultData('0731')
dataset = Dataset(data)
usedLayers = ['ndvi', 'slope', 'aspect']
dataset.usedLayers = usedLayers
trainData, testData = dataset.split()

inp = trainData.getData()
weather, aois, out = inp
print(aois.shape)

# inputForModel = data.getAOIs(usedLayers, trainIndices)
# inputForTest = data.getAOIs(usedLayers, testIndices)
# # print('trainind shape ', trainIndices.size)
# nchannels = trainIndices[1]
# print(trainIndices[1])
# print(trainIndices[2])
# print(trainIndices[3])
# aoisize = [trainIndices[2],trainIndices[3]]

# cnnModel = lib.model.ImageBranch(nchannels, aoisize)
# ccnModel.fit(inputForModel)



# print('aois shape ', aois.shape)
nchannels = aois.shape[1]
print('THIS IS INPUT: ', aois.shape)
aoisize = [aois.shape[2], aois.shape[3]] #

cnnModel = lib.model.Model(weather.shape[0],nchannels, aoisize)




# predictions_test = m.predict(trainData)
# print(predictions_test)

# testData = data[testIndices]
# predictions_train = m.predict(testData)
# print(predictions_train)

# np.savetxt('output/predictions_test.csv', predictions_test, delimiter = ',')
# np.savetxt('output/predictions_train.csv', predictions_train, delimiter = ',')

# res = viz.reassemblePredictions(predictions_test, trainIndices, data.shape)
# viz.show(res)
# evaluate the model
# scores = m.evaluate(test_today, test_tomorrow)
# print("\n%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
