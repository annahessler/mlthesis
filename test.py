#test.py
import unittest

from hottopic import rawdata
from hottopic import dataset

class TestRawdata(unittest.TestCase):

    # def setUp(self):
    #     self.

    def test_load(self):
        self.raw = rawdata.load()
        # print("loaded the RawData:" + str(self.raw))

    def test_findAvailable(self):
        # print("found these burns and dates:")
        for b in rawdata.availableBurns():
            # print(b + ":")
            for d in rawdata.availableDates(b):
                # print('\t' + d)
                pass

class TestDataset(unittest.TestCase):

    def test_default(self):
        self.ds = dataset.load()

    def test_saveLoad(self):
        self.ds = dataset.load()
        FNAME = 'output/datasets/test.npz'
        self.ds.save(FNAME)
        reloaded = dataset.load(FNAME)
        import os
        os.remove(FNAME)
        self.assertEqual(reloaded, self.ds)
        # print(reloaded)
        # print(self.ds)

if __name__ == '__main__':
    unittest.main()
