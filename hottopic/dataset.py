import math
from collections import namedtuple
import random
import json
from time import localtime, strftime

import numpy as np
import cv2
# from rawdata import
from hottopic import rawdata
from hottopic import viz
from keras.preprocessing.image import ImageDataGenerator

def load(fname=None):
    if fname is None:
        # give us the default dataset of everything
        return Dataset(rawdata.load())
    fname = fixFileName(fname)
    with np.load(fname) as archive:
        # np.load gives us back a weird structure.
        # we need structure of {burnName:{date:nparray}}
        print('loaded the dataset file {}'.format(fname))
        d = dict(archive)
        print('converted to dict')
        pointList = {}
        for burnName in d:
            print('opening burn {}'.format(burnName))
            pointList[burnName] = d[burnName][()]

        print('creating dataset...')
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
            print('loading rawdata...')
            data = rawdata.load()
        self.data = data
        print('decoding points...')
        self.masks = self._decodePoints(points)


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
        if burnName not in self.masks:
            self.masks[burnName] = {}
        if date in self.masks[burnName]:
            return #nothing to do
        perim = self.data.burns[burnName].days[date].startingPerim
        mask = np.ones_like(perim, dtype=np.uint8)
        self.masks[burnName][date] = mask

    def remove(self, burnName, date):
        if burnName not in self.masks:
            raise ValueError("burnName {} is not in this dataset".format(burnName))
        if date not in self.masks[burnName]:
            raise ValueError("date {} is not in this dataset.masks[{}]".format(date, burnName))
        del self.masks[burnName][date]
        if len(self.masks[burnName]) == 0:
            del self.masks[burnName]

    def copy(self):
        '''the underlying RawData object doesn't need to be copied,
        but the dict of masks does, since the masks may change'''
        newPoints = {}
        for burnName in self.masks:
            burn = self.masks[burnName]
            d = {}
            for date in burn:
                d[date] = burn[date].copy()
            newPoints[burnName] = d
        return Dataset(self.data, newPoints)

    def getUsedBurnNamesAndDates(self):
        for name in self.masks.keys():
            for date in self.masks[name].keys():
                yield (name,date)

    def getAllLayers(self, layerName):
        result = {}
        allBurnNames = self.masks.keys()
        for burnName in allBurnNames:
            burn = self.data.burns[burnName]
            layer = burn.layers[layerName]
            result[burnName] = layer
        return result

    def getDays(self):
        for burnName, dayDict in self.masks.items():
            for date in dayDict:
                yield self.data.burns[burnName].days[date]

    def getDaysAndMasks(self):
        for burnName, dayDict in self.masks.items():
            for date, pointMask in dayDict.items():
                yield self.data.burns[burnName].days[date], pointMask

    def getPoints(self):
        for burnName, dayDict in self.masks.items():
            for date, pointMask in dayDict.items():
                w = np.where(pointMask)
                yield (burnName, date, w)

    def save(self, fname=None):
        if fname is None:
            fname = strftime("%d%b%H-%M", localtime())
        fname = fixFileName(fname)
        np.savez_compressed(fname, **self.masks)

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
            oldMask = self.masks[burnName][date]
            # get every location that satisfies the condition
            day = self.data.burns[burnName].days[date]
            newMask = filterFunction(day, **kwargs)
            oldMask = oldMask.astype(np.uint8)
            newMask = newMask.astype(np.uint8)
            anded = np.bitwise_and(oldMask, newMask)
            self.masks[burnName][date] = anded

    def evenOutPositiveAndNegative(self):
        for burnName, date in self.getUsedBurnNamesAndDates():
            day = self.data.getDay(burnName, date)
            burned = day.endingPerim.astype(np.uint8)
            didNotBurn = 1-burned
            currentMask = self.masks[burnName][date].astype(np.uint8)
            # all the pixels we are training on that DID and did NOT burn
            pos = np.bitwise_and(burned, currentMask)
            neg = np.bitwise_and(didNotBurn, currentMask)

            numPos = np.count_nonzero(pos)
            numNeg = np.count_nonzero(neg)
            if numPos > numNeg:
                idxs = np.where(pos.flatten())[0]
                numToZero = numPos-numNeg
            else:
                idxs = np.where(neg.flatten())[0]
                numToZero = numNeg-numPos
            if len(idxs) == 0:
                continue
            toBeZeroed = np.random.choice(idxs, numToZero)
            origShape = currentMask.shape
            currentMask = currentMask.flatten()
            currentMask[toBeZeroed] = 0
            currentMask = currentMask.reshape(origShape)
            self.masks[burnName][date] = currentMask

    def evenOutPositiveAndNegativeOld(self):
        '''Make it so our dataset is a more even mixture of yes and no outputs'''
        # yes will contain all 'did burn' points, no contains 'did not burn' points
        yes = []
        no =  []
        for p in self.toList(self.masks):
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
    #     for burnName, dayDict in self.masks.items():
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
        startingPerim = day.startingPerim.astype(np.uint8)
        kernel = np.ones((3,3))
        its = int(round((2*(radius/rawdata.PIXEL_SIZE)**2)**.5))
        dilated = cv2.dilate(startingPerim, kernel, iterations=its)
        border = dilated - startingPerim
        return border.astype(np.uint8)

    def __len__(self):
        total = 0
        for dayDict in self.masks.values():
            for series in dayDict.values():
                total += series.shape[1]
        return total

    def __eq__(self, other):
        if isinstance(other, Dataset):
            return self.masks == other.masks
        else:
            return NotImplemented

    def __repr__(self):
        # shorten the string repr of self.masks
        return "Dataset({}, with {} points)".format(self.data, len(self))
