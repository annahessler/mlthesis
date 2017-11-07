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
            filterFunc = points
            self.points = self.filterPoints(self.data, filterFunc)

        assert type(self.points) == type({})

    def getUsedBurnNamesAndDates(self):
        results = []
        burnNames = self.points.keys()
        for name in burnNames:
            dates = self.points[name].keys()
            for date in dates:
                results.append((name,date))
        return results

    def getAllLayers(self, layerName):
        result = {}
        allBurnNames = list(self.points.keys())
        for burnNamename in allBurnNames:
            burn = self.data.burns[burnName]
            layer = burn.layers[layerName]
            result[burnName] = layer
        return result

    @staticmethod
    def toList(pointDict):
        '''Flatten the point dictionary to a list of Points'''
        result = []
        for burnName in pointDict:
            dayDict = pointDict[burnName]
            for date in dayDict:
                points = dayDict[date]
                result.extend(points)
        return result

    @staticmethod
    def toDict(pointList):
        burns = {}
        for p in pointList:
            burnName, date, location = p
            if burnName not in burns:
                burns[burnName] = {}
            if date not in burns[burnName]:
                burns[burnName][date] = []
            if p not in burns[burnName][date]:
                burns[burnName][date].append(p)
        return burns

    @staticmethod
    def filterPoints(data, filterFunction):
        '''Return all the points which satisfy some filterFunction'''
        points = {}
        burns = data.burns.values()
        for b in burns:
            dictOfDays = {}
            points[b.name] = dictOfDays
            days = b.days.values()
            for d in days:
                # get every location that satisfies the condition
                locations = filterFunction(b, d)
                dictOfDays[d.date] = [Point(b.name,d.date,l) for l in locations]
        return points

    def evenOutPositiveAndNegative(self):
        '''Make it so our dataset is a more even mixture of yes and no outputs'''
        # yes will contain all 'did burn' points, no contains 'did not burn' points
        yes = []
        no =  []
        for p in self.toList():
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
        return self.toDict(yes+no)

    def split(self, ratios=[.5]):
        ptList = self.toList(self.points)
        random.shuffle(ptList)
        beginIndex = 0
        ratios.append(1)
        sets = []
        for r in ratios:
            endIndex = int(round(r * len(ptList)))
            # print(beginIndex, endIndex)
            newPts = self.toDict(self.points[beginIndex:endIndex])
            sets.append( Dataset(self.data, newPts))
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
        return "Dataset({}, with {} points)".format(self.data, len(self.toList(self.points)))

# create a class that represents a spatial and temporal location that a sample lives at
Point = namedtuple('Point', ['burnName', 'date', 'location'])

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
