
import numpy as np
import lib.model
import lib.data

dataset = lib.data.Dataset.defaultDataset('0731')
# print(dataset)
# print(dataset.getInputs())

trainIndices, testIndices = lib.data.chooseDatasets(dataset)
usedLayers = ['slope', 'ndvi']
inputForModel = dataset.getAOIs(usedLayers, trainIndices)
inputForTest = dataset.getAOIs(usedLayers, testIndices)
# print('trainind shape ', trainIndices.size)
nchannels = inputForModel.shape[1]
print('THIS IS INPUT: ', inputForModel.shape)
aoisize = [inputForModel.shape[2], inputForModel.shape[3]]

cnnModel = lib.model.ImageBranch(nchannels, aoisize)
cnnModel.fitModel(inputForModel, inputForTest)
# print(trainIndices)

# m = lib.model.Model(trainData)
# m.fit(trainData)

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
