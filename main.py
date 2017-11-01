
import numpy as np

from lib import model
from lib import datamodule
from lib import dataset
from lib import viz
from lib import histories

data = datamodule.Data.defaultData('0731')
dataset = dataset.Dataset(data)

usedLayers = ['perim', 'ndvi', 'slope', 'aspect']
dataset.usedLayers = usedLayers
dataset.usedWeather = []
#dataset.usedWeather = ['maxTemp','avgHum']
trainData, validateData, testData = dataset.split()

print('testdata is ' , testData)

mod = model.FireModel(*trainData.getModelParams())
mod.fit(trainData, validateData, testData)
# print(mod.history.losses)
predictions = mod.predict(testData)
np.savetxt("res.csv", predictions, delimiter=',')

# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

res = viz.visualizePredictions(testData, predictions)
viz.show(res)
viz.save(res,'predictions')
