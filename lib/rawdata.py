
import numpy as np
import cv2

from lib import util

PIXEL_SIZE = 30
_memoedAllBurns = None



def load(burnNames='all', dates='all'):
    global _memoedAllBurns
    if _memoedAllBurns and burnNames=='all' and dates=='all':
        return _memoedAllBurns
    if burnNames == 'all':
        burnNames = util.listdir_nohidden('data/')
    if dates == 'all':
        burns = {n:Burn.load(n, 'all') for n in burnNames}
    else:
        # assumes dates is a dict, with keys being burnNames and vals being dates
        burns = {n:Burn.load(n, dates[n]) for n in burnNames}
    result = RawData(burns)
    if burnNames=='all' and dates=='all':
        _memoedAllBurns = result
    return result

class RawData(object):

    def __init__(self, burns):
        self.burns = burns

    def getWeather(self, burnName, date):
        burn = self.burns[burnName]
        day = burn.days[date]
        return day.weather

    def getOutput(self, burnName, date, location):
        burn = self.burns[burnName]
        day = burn.days[date]
        return day.endingPerim[location]

    def getDay(self, burnName, date):
        return self.burns[burnName].days[date]

    def __repr__(self):
        bs = []
        for name, burn in sorted(self.burns.items()):
            bs.append( name + ':' + str(sorted(burn.days.keys())) )
        return "RawData({})".format(', '.join(bs))

class Burn(object):

    def __init__(self, name, days=None, layers=None):
        self.name = name
        self.days = {} if days is None else days
        self.layers = layers if layers is not None else self.loadLayers()

        # what is the height and width of a layer of data
        self.layerSize = list(self.layers.values())[0].shape[:2]

    def loadLayers(self):
        folder = 'data/{}/'.format(self.name)
        dem = util.openImg(folder+'dem.tif')
        # slope = util.openImg(folder+'slope.tif')
        band_2 = util.openImg(folder+'band_2.tif')
        band_3 = util.openImg(folder+'band_3.tif')
        band_4 = util.openImg(folder+'band_4.tif')
        band_5 = util.openImg(folder+'band_5.tif')
        ndvi = util.openImg(folder+'ndvi.tif')
        aspect = util.openImg(folder+'aspect.tif')
        # r,g,b,nir = cv2.split(landsat)

        layers = {'dem':dem,
                # 'slope':slope,
                'ndvi':ndvi,
                'aspect':aspect,
                'band_4':band_4,
                'band_3':band_3,
                'band_2':band_2,
                'band_5':band_5}

        # ok, now we have to make sure that all of the NoData values are set to 0
        #the NV pixels occur outside of our AOIRadius
        # When exported from GIS they could take on a variety of values
        # susceptible = ['dem', 'r','g','b','nir',]
        #TODO
        for name, layer in layers.items():
            pass
        return layers

    @staticmethod
    def load(burnName, dates='all'):
        burn = Burn(burnName)
        if dates == 'all':
            dates = util.availableDates(burnName)
        days = {date:Day(burn, date) for date in dates}
        burn.days = days
        return burn

    def __repr__(self):
        return "Burn({}, {})".format(self.name, [d.date for d in self.days.values()])

class Day(object):

    def __init__(self, burn, date, weather=None, startingPerim=None, endingPerim=None):
        self.burn = burn
        self.date = date
        self.weather = weather             if weather       is not None else self.loadWeather()
        self.startingPerim = startingPerim if startingPerim is not None else self.loadStartingPerim()
        self.endingPerim = endingPerim     if endingPerim   is not None else self.loadEndingPerim()

    def loadWeather(self):
        fname = 'data/{}/weather/{}.csv'.format(self.burn.name, self.date)
        # the first row is the headers, and only cols 4-11 are actual data
        data = np.loadtxt(fname, skiprows=1, usecols=range(5,12), delimiter=',').T
        # now data is 2D array
        return data

    def loadStartingPerim(self):
        fname = 'data/{}/perims/{}.tif'.format(self.burn.name, self.date)
        perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if perim is None:
            raise RuntimeError('Could not find a perimeter for the fire {} for the day {}'.format(self.burn.name, self.date))
        perim[perim!=0] = 255
        return perim

    def loadEndingPerim(self):
        guess1, guess2 = util.possibleNextDates(self.date)
        fname = 'data/{}/perims/{}.tif'.format(self.burn.name, guess1)
        perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if perim is None:
            # overflowed the month, that file didnt exist
            fname = 'data/{}/perims/{}.tif'.format(self.burn.name, guess2)
            perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            if perim is None:
                raise RuntimeError('Could not open a perimeter for the fire {} for the day {} or {}'.format(self.burn.name, guess1, guess2))
        return perim

    def __repr__(self):
        return "Day({},{})".format(self.burn.name, self.date)
