
import numpy as np
import lib.model
import lib.data
import lib.viz as viz

data = lib.data.createData('0731')
trainIndices, testIndices = lib.data.chooseDatasets(data)
trainData = data[trainIndices]

m = lib.model.Model(trainData)
m.fit(trainData)

predictions_test = m.predict(trainData)
print(predictions_test)

testData = data[testIndices]
predictions_train = m.predict(testData)
print(predictions_train)

np.savetxt('output/predictions_test.csv', predictions_test, delimiter = ',')
np.savetxt('output/predictions_train.csv', predictions_train, delimiter = ',')

res = viz.reassemblePredictions(predictions_test, trainIndices, data.shape)
viz.show(res)
# evaluate the model
# scores = m.evaluate(test_today, test_tomorrow)
# print("\n%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
