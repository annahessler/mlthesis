import numpy as np
import cv2
from .datamodule import PIXEL_SIZE

class Dataset(object):

    def __init__(self, data, whichPixels=None, whichBurns=None, whichDays=None, whichLayers=None, weatherMetrics = None):
        self.data = data
        self.pixels = whichPixels
        self.burns = whichBurns
        self.days = whichDays
        self.layers = whichLayers #this will be kept track of in main...wont change
        self.metrics = weatherMetrics #this will be created in another class, kept track of in main...wont change

    def getData(self):
        return self.data #return input/output?

    def getPixels(self):
        return self.pixels

    def getBurns(self):
        return self.burns

    def getDays(self):
        return self.days

    def getLayers(self):
        return self.layers
