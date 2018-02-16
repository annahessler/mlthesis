#test.py
import unittest

from lib import rawdata
from lib import dataset

class TestRawdata(unittest.TestCase):

    # def setUp(self):
    #     self.

    def test_load(self):
        self.raw = rawdata.load()

    def test_findAvailable(self):
        print("found these burns and dates:")
        for b in rawdata.availableBurns():
            print(b + ":")
            for d in rawdata.availableDates(b):
                print('\t' + d)

class TestDataset(unittest.TestCase):


    @unittest.skip("takes too long")
    def test_defualt(self):
        self.ds = dataset.load()

if __name__ == '__main__':
    unittest.main()
