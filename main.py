
import numpy as np

# from lib import model
from lib import datamodule
from lib import dataset
from lib import viz

data = datamodule.Data.defaultData('0731')
dataset = dataset.Dataset(data)

usedLayers = ['perim', 'ndvi', 'slope', 'aspect']
dataset.usedLayers = usedLayers
dataset.usedWeather = []
#dataset.usedWeather = ['maxTemp','avgHum']
trainData, testData = dataset.split()

weather, aois, out = trainData.getData()
nsamples, height, width, nchannels = aois.shape

mod = lib.model.FireModel(weather.shape[1], nchannels, (height, width))
mod.fit(trainData)
res = mod.predict(testData)
np.savetxt("res.csv", res, delimiter=',')

# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

res = viz.visualizePredictions(testData, predictions)
viz.show(res)
viz.save(res,'predictions')
