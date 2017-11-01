import numpy as np
import cv2
from .datamodule import PIXEL_SIZE

class Dataset(object):

    def __init__(self, data, indices=None, usedLayers=None, usedWeather=None):
        self.data = data
        if indices is None:
            # use every pixel
            i = np.array(list(np.ndindex(self.data.shape)))
            xs, ys = i[:,0], i[:,1]
            self.indices = (xs, ys)
        else:
            self.indices = indices

        self.usedLayers = usedLayers
        self.usedWeather = usedWeather

        self.VULNERABLE_RADIUS = 300
        self.KERNEL_RADIUS     = 15

    def getAOIs(self, radius=None):
        if radius is None:
            radius = self.KERNEL_RADIUS
        stacked = self.data.stackLayers(self.usedLayers)
        aois = []
        r = radius
        for y,x in zip(*self.indices):
            aois.append(stacked[y-r:y+r+1,x-r:x+r+1,:])
        arr = np.array(aois)
        return arr

    def getWeather(self):
        weather = self.data.stackWeather(self.usedWeather)
        nsamples = len(self.indices[0])
        return np.tile(weather, (nsamples,1))

    def getOutput(self):
        return self.data.output[self.indices]

    def getData(self):
        inputs =  [self.getWeather(), self.getAOIs()]
        output = self.getOutput()
        return inputs, output

    def getLayer(self, layerName, which='used'):
        l = self.data.layers[layerName]
        if which == 'used':
            res = np.zeros_like(l)
            res[self.indices] = l[self.indices]
            return res
        elif which == 'all':
            return l
        else:
            raise ValueError("Expected 'used' or 'all' as argument for parameter which")

    def findVulnerablePixels(self, startingPerim, radius=None):
        if radius is None:
            radius = self.VULNERABLE_RADIUS
        '''Return the indices of the pixels that close to the current fire perimeter'''
        kernel = np.ones((3,3))
        its = int(round((2*(radius/PIXEL_SIZE)**2)**.5))
        dilated = cv2.dilate(startingPerim, kernel, iterations=its)
        border = dilated - startingPerim
        return np.where(border)

    def split(self, trainRatio=.60, validateRatio=.10, shuffle=True):
        startingPerim = self.data.layers['perim']
        xs,ys = self.findVulnerablePixels(startingPerim)
        if shuffle:
            p = np.random.permutation(len(xs))
            xs,ys = xs[p], ys[p]
        # now split them into train and test sets
        trainIndex = ti = int(trainRatio*len(xs))
        validateIndex = vi = int((trainRatio+validateRatio)*len(xs))
        trainIndices = (xs[:ti], ys[:ti])
        validateIndices  = (xs[ti:vi], ys[ti:vi])
        testIndices  = (xs[vi:], ys[vi:])

        train = Dataset(self.data, trainIndices, self.usedLayers, self.usedWeather)
        validate = Dataset(self.data, validateIndices, self.usedLayers, self.usedWeather)
        test = Dataset(self.data, testIndices, self.usedLayers, self.usedWeather)
        return train, validate, test

    def normalize(self): #this still needs to be called
        for key in self.data.layers:
            layer = self.data.layers[key].astype(np.float32)
            minimum = layer.min()
            layer = layer - minimum
            maximum = layer.max()
            layer = layer/maximum
            self.data.layers[key] = layer

    def getModelParams(self):
        kernelSize = 1+2*self.KERNEL_RADIUS
        numberWeatherEntries = len(self.usedWeather) if self.usedWeather is not None else len(self.data.weather)
        numberLayers = len(self.usedLayers) if self.usedLayers is not None else len(self.data.layers)
        return numberWeatherEntries, numberLayers, kernelSize

    def __len__(self):
        return len(self.indices[0])
