import numpy as np
import cv2

try:
    import matplotlib.pyplot as plt
except:
    pass

def openImg(fname):
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    try:
        img = img.astype(np.float32)
    except AttributeError:
        raise ValueError("Could not open the file {} as an image".format(fname))
    channels = cv2.split(img)
    for c in channels:
        c[invalidPixelIndices(c)] = np.nan
    return cv2.merge(channels)

def validPixelIndices(layer):
    validPixelMask = 1-invalidPixelMask(layer)
    return np.where(validPixelMask)

def invalidPixelIndices(layer):
    return np.where(invalidPixelMask(layer))

def invalidPixelMask(layer):
    # If there are any massively valued pixels, just return those
    HUGE = 1e10
    huge = np.absolute(layer) > HUGE
    if np.any(huge):
        return huge

    # floodfill in from every corner, all the NODATA pixels are the same value so they'll get found
    h,w = layer.shape[:2]
    noDataMask = np.zeros((h+2,w+2), dtype = np.uint8)
    fill = 1
    seeds = [(0,0), (0,h-1), (w-1,0), (w-1,h-1)]
    for seed in seeds:
        cv2.floodFill(layer.copy(), noDataMask, seed, fill)
        # plt.figure('layer')
        # plt.imshow(layer)
        # plt.figure('noDataMask')
        # plt.imshow(noDataMask)
        # plt.show()

    # extract ouf the center of the mask, which corresponds to orig image
    noDataMask = noDataMask[1:h+1, 1:w+1]
    return noDataMask

def normalize(arr, axis=None):
    '''Rescale an array so that it varies from 0-1.

    if axis=0, then each column is normalized independently
    if axis=1, then each row is normalized independently'''

    arr = arr.astype(np.float32)
    # print('subtracting min')
    res = arr - np.nanmin(arr, axis=axis)
    # print('dividing where', res)
    # where dividing by zero, just use zero
    res = np.divide(res, np.nanmax(res, axis=axis), out=np.zeros_like(res), where=res!=0)
    # print('done')
    return res

def partition(things, ratios=None):
    if ratios is None:
        ratios = [.5]
    beginIndex = 0
    ratios.append(1)
    partitions = []
    for r in ratios:
        endIndex = int(round(r * len(things)))
        # print(beginIndex, endIndex)
        section = things[beginIndex:endIndex]
        partitions.append(section)
        beginIndex = endIndex
    return partitions



def savePredictions(predictions, fname=None):
    directory = 'output/predictions/'
    if fname is None:
        timeString = strftime("%d%b%H:%M", localtime())
        fname = directory + '{}.csv'.format(timeString)
    if not fname.startswith(directory):
        fname = directory + fname
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for pt, pred in predictions.items():
            burnName, date, location = pt
            y,x = location
            row = [str(burnName), str(date), str(y), str(x), str(pred)]
            writer.writerow(row)

def openPredictions(fname):
    result = {}
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            burnName, date, y, x, pred = row
            # ensure the date is 4 chars long
            date = str(date).zfill(4)
            x = int(x)
            y = int(y)
            pred = float(pred)
            p = dataset.Point(burnName, date, (y,x))
            result[p] = pred
    return result


if __name__ == '__main__':
    import glob
    import matplotlib.pyplot as plt
    import random
    folder = 'data/**/perims/'
    types = ('*.tif', '*.png') # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(folder+files))
    # random.shuffle(files_grabbed)
    for f in files_grabbed:
        print(f)
        # if 'riceRidge' not in f:
        #     continue
        img = openImg(f)
        plt.figure(f)
        if len(img.shape) > 2 and img.shape[2] > 3:
            plt.imshow(img[:,:,:3])
        else:
            plt.imshow(img)
        plt.show()
