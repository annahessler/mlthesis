from collections import namedtuple
import random

import numpy as np
import cv2
# from rawdata import
from lib.rawdata import RawData, PIXEL_SIZE
from lib import viz
# from model import InputSettings

class Dataset(object):
    '''A set of Point objects'''
    VULNERABLE_RADIUS = 500

    def __init__(self, data, points='all'):
        self.data = data

        self.points = points
        if points=='all':
            points = Dataset.allPixels
        if hasattr(points, '__call__'):
            # points is a filter function
            self.points = self.getPoints(points)

        assert type(self.points) == type([])
        for p in self.points:
            assert type(p) == Point

    def getUsedBurnNames(self):
        used = []
        for burnName, date, location in self.points:
            if burnName not in used:
                used.append(burnName)
        return used

    def getUsedBurnNamesAndDates(self):
        used = []
        for burnName, date, location in self.points:
            if (burnName, date) not in used:
                used.append((burnName, date))
        return used

    def getPoints(self, filterFunction):
        '''Return all the points which satisfy some filterFunction'''
        points = []
        burns = self.data.burns.values()
        for b in burns:
            days = b.days.values()
            for d in days:
                # get every location that satisfies the condition
                locations = filterFunction(b, d)
                for l in locations:
                    points.append(Point(b.name,d.date,l))
        return points

    def evenOutPositiveAndNegative(self):
        '''Make it so our dataset is a more even mixture of yes and no outputs'''
        # yes will contain all 'did burn' points, no contains 'did not burn' points
        yes = []
        no =  []
        for p in self.points:
            burnName, date, loc = p
            burn = self.data.burns[burnName]
            day = burn.days[date]
            out = day.endingPerim[loc]
            if out:
                yes.append(p)
            else:
                no.append(p)
        # shorten whichever is longer
        if len(yes) > len(no):
            random.shuffle(yes)
            yes = yes[:len(no)]
        else:
            random.shuffle(no)
            no = no[:len(yes)]

        # recombine
        return yes+no

    def getSamples(self, inputSettings):
        return [Sample(self.data, p, inputSettings) for p in self.points]

    def split(self, ratios=[.5]):
        random.shuffle(self.points)
        beginIndex = 0
        ratios.append(1)
        sets = []
        for r in ratios:
            endIndex = int(round(r * len(self.points)))
            # print(beginIndex, endIndex)
            sets.append( Dataset(self.data, self.points[beginIndex:endIndex]) )
            beginIndex = endIndex
        return sets

    @staticmethod
    def allPixels(burn, day):
        return list(np.ndindex(burn.layerSize))

    @staticmethod
    def vulnerablePixels(burn, day, radius=VULNERABLE_RADIUS):
        '''Return the indices of the pixels that close to the current fire perimeter'''
        startingPerim = day.startingPerim
        kernel = np.ones((3,3))
        its = int(round((2*(radius/PIXEL_SIZE)**2)**.5))
        dilated = cv2.dilate(startingPerim, kernel, iterations=its)
        border = dilated - startingPerim
        ys, xs = np.where(border)
        return list(zip(ys, xs))

    def __repr__(self):
        # shorten the string repr of self.points
        return "Dataset({}, with {} points)".format(self.data, len(self.points))

# create a class that represents a spatial and temporal location that a sample lives at
Point = namedtuple('Point', ['burnName', 'date', 'location'])

class Sample(object):
    '''An actual sample that can be fed into the network.

    Encapsulates both the input and output data, as well as the Point it was taken from'''

    # if we calculate the weatherMetric for a Day, memo it
    memoedWeatherMetrics = {}
    memoedPadded = {}

    def __init__(self, data, point, inputSettings):
        self.data = data
        self.point = point
        self.inputSettings = inputSettings

    def getInputs(self):
        burnName, date, location = self.point
        burn = self.data.burns[burnName]
        day = burn.days[date]

        inpset = self.inputSettings
        if (burnName, date) in Sample.memoedWeatherMetrics:
            weatherMetrics = Sample.memoedWeatherMetrics[(burnName, date)]
        else:
            weatherMetrics = inpset.weatherMetrics.calculate(day.weather)
            Sample.memoedWeatherMetrics[(burnName, date)] = weatherMetrics

        if (burnName, date) not in Sample.memoedPadded:
            padded = self.stackAndPadLayers(burn.layers, day.startingPerim)
            Sample.memoedPadded[(burnName, date)] = padded
        else:
            padded = Sample.memoedPadded[(burnName, date)]
        aoi = self.extract(padded, location)

        return [weatherMetrics, aoi]

    def extract(self, padded, location):
        '''Assume padded is bordered by radius self.inputSettings.AOIRadius'''
        y,x = location
        r = self.inputSettings.AOIRadius
        lox = r+(x-r)
        hix = r+(x+r+1)
        loy = r+(y-r)
        hiy = r+(y+r+1)
        aoi = padded[loy:hiy,lox:hix]
        # print(stacked.shape, padded.shape)s
        return aoi

    def stackAndPadLayers(self, layers, startingPerim):
        usedLayerNames, metric, AOIRadius = self.inputSettings
        if usedLayerNames == 'all':
            names = list(layers.keys())
            names.sort()
            usedLayers = [layers[name] for name in names]
        else:
            usedLayers = [layers[name] for name in usedLayerNames]
        usedLayers.insert(0, startingPerim)
        stacked = np.dstack(usedLayers)

        r = AOIRadius
        # pad with zeros around border of image
        padded = np.lib.pad(stacked, ((r,r),(r,r),(0,0)), 'constant')
        return padded

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
    masterDataSet = Dataset(d, points=Dataset.vulnerablePixels)
    print(masterDataSet)
    masterDataSet.points = masterDataSet.evenOutPositiveAndNegative()
    print(masterDataSet)
    train, validate, test = masterDataSet.split(ratios=[.6,.7])
    print(train)
    print(validate)
    print(test)
    p = Point('riceRidge', '0731', (20, 156))
    # print(p)
    inpset = InputSettings(['b'], AOIRadius=60)
    s = Sample(d, p, inpset)
    print(s)
    w, aoi = s.getInputs()
    startPerim = aoi[:,:,1]
    viz.show(img, startPerim)
    # viz.show()

    print(s.getOutput())
