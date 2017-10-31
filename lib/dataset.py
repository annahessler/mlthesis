import numpy as np
import cv2
from .datamodule import PIXEL_SIZE

VULNERABLE_RADIUS = 300

class Dataset(object):

    def __init__(self, data, indices=None, usedLayers=None, usedWeather=None):
        self.data = data
        if indices is None:
            # use every pixel
            self.indices = np.ndindex(self.data.shape)
        else:
            self.indices = indices

        self.usedLayers = usedLayers
        self.usedWeather = usedWeather

    def getAOIs(self, radius=15):
        stacked = self.data.stackLayers(self.usedLayers)
        aois = []
        r = radius
        for y,x in zip(*self.indices):
            aois.append(stacked[y-r:y+r+1,x-r:x+r+1,:])
        arr = np.array(aois)

        # we need the tensor to have dimensions (nsamples, nchannels, AOIwidth,AOIheight)
        # it starts as (nsamples, AOIwidth,AOIheight, nchannels)
        # arr = np.swapaxes(arr,1,3)
        # arr = np.swapaxes(arr,2,3)
        return arr

    def getWeather(self):
        weather = self.data.stackWeather(self.usedWeather)
        nsamples = len(self.indices[0])
        return np.tile(weather, (nsamples,1))

    def getOutput(self):
        return self.data.output[self.indices]

    def getData(self):
        return self.getWeather(), self.getAOIs(), self.getOutput()

    @staticmethod
    def findVulnerablePixels(startingPerim, radius=VULNERABLE_RADIUS):
        '''Return the indices of the pixels that close to the current fire perimeter'''
        kernel = np.ones((3,3))
        its = int(round((2*(radius/PIXEL_SIZE)**2)**.5))
        dilated = cv2.dilate(startingPerim, kernel, iterations=its)
        border = dilated - startingPerim
        return np.where(border)

    def split(self, ratio=.75, shuffle=True):
        startingPerim = self.data.layers['perim']
        xs,ys = Dataset.findVulnerablePixels(startingPerim)
        if shuffle:
            p = np.random.permutation(len(xs))
            xs,ys = xs[p], ys[p]
        # now split them into train and test sets
        splitIndex = si = int(ratio*len(xs))
        trainIndices = (xs[:si], ys[:si])
        testIndices  = (xs[si:], ys[si:])

        train = Dataset(self.data, trainIndices, self.usedLayers, self.usedWeather)
        test = Dataset(self.data, testIndices, self.usedLayers, self.usedWeather)
        return train, test

    def normalize(self): #this still needs to be called
        for key in self.data.layers:
            layer = self.data.layers[key].astype(np.float32)
            minimum = min(layer)
            layer = layer - minimum
            maximum = max(layer)
            layer = layer/maximum
            self.data.layers[key] = layer
