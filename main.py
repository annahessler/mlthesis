
import numpy as np
# import matplotlib.pylab as plt
import lib.model
from lib.datamodule import Data
from lib.dataset import Dataset



data = Data.defaultData('0731')
dataset = Dataset(data)
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

print(res)
