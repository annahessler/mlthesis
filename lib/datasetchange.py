from collections import namedtuple

import numpy as np
import cv2
# from datamodule import PIXEL_SIZE
from rawdata import RawData
import viz
# from model import InputSettings

class Dataset(object):
    '''A set of Point objects'''
    VULNERABLE_RADIUS = 500

    def __init__(self, data, points='all'):
        self.data = data

        if points=='all':
            points = self.allPoints()
        self.points = points

        assert type(self.points) == type([])
        for p in self.points:
            assert type(p) == Point

    def allPoints(self):
        samples = []
        burns = self.data.burns.values()
        for b in burns:
            days = b.days.values()
            for d in days:
                # get every location in the layer of data
                locations = list(np.ndindex(b.layerSize))
                for l in locations:
                    samples.append(Point(b.name,d.date,l))
        return samples

    def findVulnerablePixels(self, radius=None):
        if radius is None:
            radius = self.VULNERABLE_RADIUS
        '''Return the indices of the pixels that close to the current fire perimeter'''
        kernel = np.ones((3,3))
        its = int(round((2*(radius/PIXEL_SIZE)**2)**.5))
        dilated = cv2.dilate(startingPerim, kernel, iterations=its)
        border = dilated - startingPerim
        return np.where(border)

    def getSamples(self, inputSettings):
        return [Sample(self.data, p, inputSettings) for p in self.points]

    def __repr__(self):
        # shorten the string repr of self.points
        pstring = str(self.points[:2])[:-1] + ',....,' + str(self.points[-2:])[1:]
        return "Dataset({},{})".format(self.data, pstring)

Point = namedtuple('Point', ['burnName', 'date', 'location'])

class Sample(object):
    '''An actual sample that can be fed into the network.

    Encapsulates both the input and output data, as well as the Point it was taken from'''

    def __init__(self, data, point, inputSettings):
        self.data = data
        self.point = point
        self.inputSettings = inputSettings

    def getInputs(self):
        burnName, date, location = self.point
        burn = self.data.burns[burnName]
        day = burn.days[date]

        inpset = self.inputSettings
        weatherMetrics = inpset.weatherMetrics(day.weather)

        stacked = self.stackLayers(burn.layers, day.startingPerim)
        aoi = self.getAOI(stacked, location)

        return [weatherMetrics, aoi]

    def getAOI(self, stacked, location):
        '''Pull out the aoi around a certain location from the nchannel image stacked'''
        y,x = location
        r = self.inputSettings.AOIRadius
        # pad with zeros around border of image
        padded = np.lib.pad(stacked, ((r,r),(r,r),(0,0)), 'constant')
        lox = r+(x-r)
        hix = r+(x+r+1)
        loy = r+(y-r)
        hiy = r+(y+r+1)
        aoi = padded[loy:hiy,lox:hix]
        # print(stacked.shape, padded.shape)s
        return aoi

    def stackLayers(self, layers, startingPerim):
        inpset = self.inputSettings
        if inpset.usedLayerNames == 'all':
            names = list(layers.keys())
            names.sort()
            usedLayers = [layers[name] for name in names]
        else:
            usedLayers = [layers[name] for name in inpset.usedLayerNames]
        usedLayers.insert(0, startingPerim)
        stacked = np.dstack(usedLayers)
        return stacked

    def getOutput(self):
        burnName, date, (y, x) = self.point
        burn = self.data.burns[burnName]
        day = burn.days[date]
        return day.endingPerim[y, x]

    def __repr__(self):
        return "Sample({}, {}, {})".format(self.data, self.point, self.inputSettings)

class InputSettings(object):

    def __init__(self, usedLayerNames, AOIRadius=30, weatherMetrics=None, ):
        self.AOIRadius = AOIRadius
        self.weatherMetrics = weatherMetrics if weatherMetrics is not None else InputSettings.dummyMetric
        self.usedLayerNames = usedLayerNames
        assert type(self.usedLayerNames) == list
        assert len(usedLayerNames) > 0

    @staticmethod
    def dummyMetric(weatherMatrix):
        return 42

if __name__ == '__main__':
    d = RawData.load()
    img = d.burns['riceRidge'].layers['b']
    masterDataSet = Dataset(d, points='all')
    p = Point('riceRidge', '0731', (160, 156))
    # print(p)
    inpset = InputSettings(['b'], AOIRadius=60)
    s = Sample(d, p, inpset)
    print(s)
    w, aoi = s.getInputs()
    startPerim = aoi[:,:,1]
    viz.show(img, startPerim)
    # viz.show()

    print(s.getOutput())
