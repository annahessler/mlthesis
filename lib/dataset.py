import math
from collections import namedtuple
import random
import json
from time import localtime, strftime

import numpy as np
import cv2
# from rawdata import
from lib import rawdata
from lib import viz
from keras.preprocessing.image import ImageDataGenerator
# from model import InputSettings

# create a class that represents a spatial and temporal location that a sample lives at
Point = namedtuple('Point', ['burnName', 'date', 'location'])

# def load(fname=None):
#     if fname is None:
#         # give us the default dataset of everything
#         return Dataset(rawdata.load())
#     with open(fname, 'r') as fp:
#         data = rawdata.RawData.load()
#         pts = json.load(fp)
#         newBurnDict = {}
#         for burnName, dayDict in pts.items():
#             newDayDict = {}
#             for date, ptList in dayDict.items():
#                 newPtList = [Point(name, date, tuple(loc)) for name, date, loc in ptList]
#                 newDayDict[date] = newPtList
#             newBurnDict[burnName] = newDayDict
#         # pts = [Point(name, date, tuple(loc)) for name, date, loc in pts]
#         # print(pts)
#         return Dataset(data, newBurnDict)

def load(fname=None):
    if fname is None:
        # give us the default dataset of everything
        return Dataset(rawdata.load())
    fname = fixFileName(fname)
    with np.load(fname) as archive:
        # np.load gives us back a weird structure.
        # we need structure of {burnName:{date:nparray}}
        d = dict(archive)
        pointList = {burnName:d[burnName][()] for burnName in d}
        return Dataset(data=None, points=pointList)

def emptyDataset(rawdata):
    return Dataset(data=rawdata, points={})

def fixFileName(fname):
    if not fname.endswith('.npz'):
        fname = fname + '.npz'
    return fname

class Dataset(object):
    '''A set of Point objects'''
    VULNERABLE_RADIUS = 500

    def __init__(self, data=None, points='all'):
        if data is None:
            data = rawdata.load()
        self.data = data
        self.points = self._decodePoints(points)


    def _decodePoints(self, points):
        '''Attempt to decode an input into the form of
        {str: {str:(nparray, nparray)}}, representing
        {burnName:{date:(xseries, yseries)}}'''
        if type(points) == str and points == 'all':
            points = {burnName:'all' for burnName in self.data.burns}
        assert type(points) == dict, 'expected "all" or a dictionary for burns'
        for burnName, dateDict in points.items():
            assert burnName in self.data.burns, 'Could not find burn {} in RawData {}'.format(burnName, self.data)
            if type(dateDict) == str and dateDict == 'all':
                dateDict = {date:'all' for date in self.data.burns[burnName].days}
                points[burnName] = dateDict
            for date, mask in dateDict.items():
                assert date in self.data.burns[burnName].days, 'Could not find date {} in self.data.burns[{}].days'.format(date, burnName)
                if type(mask) == str and mask == 'all':
                    perim = self.data.burns[burnName].days[date].startingPerim
                    mask = np.ones_like(perim, dtype=np.uint8)
                    dateDict[date] = mask
        return points

    def add(self, burnName, date):
        assert burnName in self.data.burns, 'Could not find burn {} in RawData {}'.format(burnName, self.data)
        assert date in self.data.burns[burnName].days, 'Could not find date {} in self.data.burns[{}].days'.format(date, burnName)
        if burnName not in self.points:
            self.points[burnName] = {}
        if date in self.points[burnName]:
            return #nothing to do
        perim = self.data.burns[burnName].days[date].startingPerim
        mask = np.ones_like(perim, dtype=np.uint8)
        self.points[burnName][date] = mask

    def remove(self, burnName, date):
        if burnName not in self.points:
            raise ValueError("burnName {} is not in this dataset".format(burnName))
        if date not in self.points[burnName]:
            raise ValueError("date {} is not in this dataset.points[{}]".format(date, burnName))
        del self.points[burnName][date]
        if len(self.points[burnName]) == 0:
            del self.points[burnName]

    def copy(self):
        '''the underlying RawData object doesn't need to be copied,
        but the dict of masks does, since the masks may change'''
        newPoints = {}
        for burnName in self.points:
            burn = self.points[burnName]
            d = {}
            for date in burn:
                d[date] = burn[date].copy()
            newPoints[burnName] = d
        return Dataset(self.data, newPoints)

    def getUsedBurnNamesAndDates(self):
        for name in self.points.keys():
            for date in self.points[name].keys():
                yield (name,date)

    def getAllLayers(self, layerName):
        result = {}
        allBurnNames = self.points.keys()
        for burnName in allBurnNames:
            burn = self.data.burns[burnName]
            layer = burn.layers[layerName]
            result[burnName] = layer
        return result

    def getDays(self):
        for burnName, dayDict in self.points.items():
            for date in dayDict:
                yield self.data.burns[burnName].days[date]

    def getDaysAndMasks(self):
        for burnName, dayDict in self.points.items():
            for date, pointMask in dayDict.items():
                yield self.data.burns[burnName].days[date], pointMask

    def getPoints(self):
        for burnName, dayDict in self.points.items():
            for date, pointMask in dayDict.items():
                w = np.where(pointMask)
                yield (burnName, date, w)

    def save(self, fname=None):
        if fname is None:
            fname = strftime("%d%b%H-%M", localtime())
        fname = fixFileName(fname)
        print('retinrg tot save to ', fname)
        np.savez_compressed(fname, **self.points)

    # @staticmethod
    # def toList(pointDict):
    #     '''Flatten the point dictionary to a list of Points'''
    #     result = []
    #     for burnName in pointDict:
    #         dayDict = pointDict[burnName]
    #         for date in dayDict:
    #             points = dayDict[date]
    #             result.extend(points)
    #     return result
    #
    # @staticmethod
    # def toDict(pointList):
    #     burns = {}
    #
    #     for i, p in enumerate(pointList):
    #         print('\r[' + '-'*(i*50)//p +']')
    #         burnName, date, location = p
    #         if burnName not in burns:
    #             burns[burnName] = {}
    #         if date not in burns[burnName]:
    #             burns[burnName][date] = []
    #         if p not in burns[burnName][date]:
    #             burns[burnName][date].append(p)
    #     return burns

    def filterPoints(self, filterFunction, **kwargs):
        '''Return all the points which satisfy some filterFunction'''
        newPoints = {}
        for burnName, date in self.getUsedBurnNamesAndDates():
            oldMask = self.points[burnName][date]
            # get every location that satisfies the condition
            day = self.data.burns[burnName].days[date]
            newMask = filterFunction(day, **kwargs)
            self.points[burnName][date] = np.logical_and(oldMask, newMask)

    # def evenOutPositiveAndNegative(self):
    #     '''Make it so our dataset is a more even mixture of yes and no outputs'''
    #     # yes will contain all 'did burn' points, no contains 'did not burn' points
    #     yes = []
    #     no =  []
    #     for p in self.toList(self.points):
    #         burnName, date, loc = p
    #         burn = self.data.burns[burnName]
    #         day = burn.days[date]
    #         out = day.endingPerim[loc]
    #         if out:
    #             yes.append(p)
    #         else:
    #             no.append(p)
    #     # shorten whichever is longer
    #     if len(yes) > len(no):
    #         random.shuffle(yes)
    #         yes = yes[:len(no)]
    #     else:
    #         random.shuffle(no)
    #         no = no[:len(yes)]
    #
    #     # recombine
    #     return self.toDict(yes+no)
    #
    # def sample(self, goalNumber='max', sampleEvenly=True):
    #     assert goalNumber == 'max' or (type(goalNumber)==int and goalNumber%2==0)
    #     # map from (burnName, date) -> [pts that burned], [pts that didn't burn]
    #     day2res = self.makeDay2burnedNotBurnedMap()
    #     # print('day2res', day2res)
    #     # find the limiting size for each day
    #     limits = {day:min(len(yes), len(no)) for day, (yes, no) in day2res.items()}
    #     print(limits)
    #     if sampleEvenly:
    #         # we must get the same number of samples from each day
    #         # don't allow a large fire to have a bigger impact on training
    #         if goalNumber == 'max':
    #             # get as many samples as possible while maintaining even sampling
    #             samplesPerDay = min(limits.values())
    #             print("samplesPerDay", samplesPerDay)
    #         else:
    #             # aim for a specific number of samples and sample evenly
    #             maxSamples = (2 * min(limits.values())) * len(limits)
    #             if goalNumber > maxSamples:
    #                 raise ValueError("Not able to get {} samples while maintaining even sampling from the available {}.".format(goalNumber, maxSamples))
    #             ndays = len(limits)
    #             samplesPerDay = goalNumber/(2*ndays)
    #             samplesPerDay = int(math.ceil(samplesPerDay))
    #     else:
    #         # we don't care about sampling evenly. Larger Days will get more samples
    #         if goalNumber == 'max':
    #             # get as many samples as possible, whatever it takes
    #             samplesPerDay = 'max'
    #         else:
    #             # aim for a specific number of samples and don't enforce even sampling
    #             maxSamples = sum(limits.values()) * 2
    #             if goalNumber > maxSamples:
    #                 raise ValueError("Not able to get {} samples from the available {}.".format(goalNumber, maxSamples))
    #     # order the days from most limiting to least limiting
    #     days = sorted(limits, key=limits.get)
    #     didBurnSamples = []
    #     didNotBurnSamples = []
    #     for i, day in enumerate(days):
    #         didBurn, didNotBurn = day2res[day]
    #         random.shuffle(didBurn)
    #         random.shuffle(didNotBurn)
    #         if sampleEvenly:
    #             print('now samplesPerDay', samplesPerDay)
    #             didBurnSamples.extend(didBurn[:samplesPerDay])
    #             didNotBurnSamples.extend(didNotBurn[:samplesPerDay])
    #         else:
    #             if samplesPerDay == 'max':
    #                 nsamples = min(len(didBurn), len(didNotBurn))
    #                 didBurnSamples.extend(didBurn[:nsamples])
    #                 didNotBurnSamples.extend(didNotBurn[:nsamples])
    #             else:
    #                 samplesToGo = goalNumber/2 - len(didBurnSamples)
    #                 daysToGo = len(days)-i
    #                 goalSamplesPerDay = int(math.ceil(samplesToGo/daysToGo))
    #                 nsamples = min(goalSamplesPerDay, len(didBurn), len(didNotBurn))
    #                 didBurnSamples.extend(didBurn[:nsamples])
    #                 didNotBurnSamples.extend(didNotBurn[:nsamples])
    #
    #     # now shuffle, trim and split the samples
    #     print('length of did and no burn samples', len(didBurnSamples), len(didNotBurnSamples))
    #     random.shuffle(didBurnSamples)
    #     random.shuffle(didNotBurnSamples)
    #     if goalNumber != 'max':
    #         didBurnSamples = didBurnSamples[:goalNumber//2]
    #         didNotBurnSamples = didNotBurnSamples[:goalNumber//2]
    #     samples = didBurnSamples + didNotBurnSamples
    #     random.shuffle(samples)
    #     print(len(samples), sum(limits.values()))
    #     return samples
    #
    # def makeDay2burnedNotBurnedMap(self):
    #     result = {}
    #     for burnName, dayDict in self.points.items():
    #         for date, ptList in dayDict.items():
    #             day = self.data.getDay(burnName, date)
    #             didBurn, didNotBurn = [], []
    #             for pt in ptList:
    #                 _,_,location = pt
    #                 if day.endingPerim[location] == 1:
    #                     didBurn.append(pt)
    #                 else:
    #                     didNotBurn.append(pt)
    #             result[(burnName, date)] = (didBurn, didNotBurn)
    #     return result
    #
    #
    # @staticmethod
    # def allPixels(burn, day):
    #     return list(np.ndindex(burn.layerSize))

    @staticmethod
    def vulnerablePixels(day, radius=VULNERABLE_RADIUS):
        '''Return the indices of the pixels that close to the current fire perimeter'''
        startingPerim = day.startingPerim
        kernel = np.ones((3,3))
        its = int(round((2*(radius/rawdata.PIXEL_SIZE)**2)**.5))
        dilated = cv2.dilate(startingPerim, kernel, iterations=its)
        border = dilated - startingPerim
        return border

    def __len__(self):
        total = 0
        for dayDict in self.points.values():
            for series in dayDict.values():
                total += series.shape[1]
        return total

    def __eq__(self, other):
        if isinstance(other, Dataset):
            return self.points == other.points
        else:
            return NotImplemented

    def __repr__(self):
        # shorten the string repr of self.points
        return "Dataset({}, with {} points)".format(self.data, len(self))
