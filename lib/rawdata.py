
from os import listdir
import numpy as np
import cv2

class RawData(object):

    def __init__(self, burns):
        self.burns = burns

    @staticmethod
    def load(burnNames='all', dates='all'):
        if burnNames == 'all':
            burnNames = listdir_nohidden('data/raw/')
        if dates == 'all':
            burns = [Burn.load(n, 'all') for n in burnNames]
        else:
            # assumes dates is a dict, with keys being burnNames and vals being dates
            burns = [Burn.load(n, dates[n]) for n in burnNames]
        return RawData(burns)

class Burn(object):

    def __init__(self, name, days, layers=None):
        self.name = name
        self.days = days
        self.layers = layers if layers is not None else self.loadLayers()

    def loadLayers(self):
        folder = 'data/raw/{}/'.format(self.name)
        dem = cv2.imread(folder+'dem.tif', cv2.IMREAD_UNCHANGED)
        slope = cv2.imread(folder+'slope.tif',cv2.IMREAD_UNCHANGED)
        landsat = cv2.imread(folder+'landsat.png', cv2.IMREAD_UNCHANGED)
        ndvi = cv2.imread(folder+'NDVI_1.tif', cv2.IMREAD_UNCHANGED)
        aspect = cv2.imread(folder+'aspect.tif', cv2.IMREAD_UNCHANGED)
        r,g,b,nir = cv2.split(landsat)
        return {'dem':dem,
                'slope':slope,
                'landsat':landsat,
                'ndvi':ndvi,
                'aspect':aspect,
                'r':r,
                'g':g,
                'b':b,
                'nir':nir}

    @staticmethod
    def load(burnName, dates='all'):
        if dates == 'all':
            dates = Day.allGoodDays(burnName)
        days = [Day(burnName, date) for date in dates]
        return Burn(burnName, days)

class Day(object):

    def __init__(self, burnName, date, weather=None, startingPerim=None, endingPerim=None):
        self.burnName = burnName
        self.date = date
        self.weather = weather             if weather       is not None else self.loadWeather()
        self.startingPerim = startingPerim if startingPerim is not None else self.loadStartingPerim()
        self.endingPerim = endingPerim     if endingPerim   is not None else self.loadEndingPerim()

    def loadWeather(self):
        fname = 'data/raw/{}/weather/{}.csv'.format(self.burnName, self.date)
        # the first row is the headers, and only cols 4-11 are actual data
        data = np.loadtxt(fname, skiprows=1, usecols=range(5,12), delimiter=',').T
        return data

    def loadStartingPerim(self):
        fname = 'data/raw/{}/perims/{}.tif'.format(self.burnName, self.date)
        perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if perim is None:
            raise RuntimeError('Could not find a perimeter for the fire {} for the day {}'.format(self.burnName, self.date))
        perim[perim!=0] = 255
        return perim

    def loadEndingPerim(self):
        print('opening perim for day after', self.date)
        guess1, guess2 = Day.nextDay(self.date)
        fname = 'data/raw/{}/perims/{}.tif'.format(self.burnName, guess1)
        print(fname)
        perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if perim is None:
            # overflowed the month, that file didnt exist
            fname = 'data/raw/{}/perims/{}.tif'.format(self.burnName, guess2)
            print('first guess didnt work')
            print(fname)
            perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            if perim is None:
                raise RuntimeError('Could not open a perimeter for the fire {} for the day {} or {}'.format(self.burnName, guess1, guess2))
        return perim

    @staticmethod
    def nextDay(dateString):
        month, day = dateString[:2], dateString[2:]

        nextDay = str(int(day)+1).zfill(2)
        guess1 = month+nextDay

        nextMonth = str(int(month)+1).zfill(2)
        guess2 = nextMonth+'01'

        return guess1, guess2


    @staticmethod
    def allGoodDays(burnName):
        '''Given a fire, return a list of all dates that we can train on'''
        directory = 'data/raw/{}/'.format(burnName)

        weatherFiles = listdir_nohidden(directory+'weather/')
        weatherDays = [fname[:-len('.csv')] for fname in weatherFiles if isValid(fname)]

        perimFiles = listdir_nohidden(directory+'perims/')
        perimDays = [fname[:-len('.tif')] for fname in perimFiles if isValid]

        # we can only use days which have perimeter data on the following day
        daysWithFollowingPerims = []
        for d in perimDays:
            nextDay1, nextDay2 = Day.nextDay(d)
            if nextDay1 in perimDays or nextDay2 in perimDays:
                daysWithFollowingPerims.append(d)

        # now we have to verify that we have weather for these days as well
        daysWithWeatherAndPerims = [d for d in daysWithFollowingPerims if d in weatherDays]
        daysWithWeatherAndPerims.sort()
        return daysWithWeatherAndPerims

def isValid(imgName):
    return cv2.imread(imgName, cv2.IMREAD_UNCHANGED) is not None

def listdir_nohidden(path):
    for f in listdir(path):
        if not f.startswith('.'):
            yield f

if __name__ == '__main__':
    raw = RawData.load()
    print(raw)
