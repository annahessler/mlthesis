from time import localtime, strftime
import csv

import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
    print('Successfully imported pyplot')
except:
    print('Failed to import pyplot ')

from lib import dataset




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


def renderPredictions(dataset):
    result = np.zeros(dataset.data.shape, dtype=np.float32)
    result[dataset.indices] = dataset.predictions+1
    return result

def renderLayer(dataset, layerName, which='used'):
    '''Create an 'image' (np array) of a given layer from a dataset'''
    shape = dataset.data.shape
    assert shape is not None, "Cannot render an empty dataset"
    # result = np.zeros(shape, dtype = np.uint
    layer = dataset.getLayer(layerName, which=which)
    normed = normalize(layer)
    return (normed*255).astype(np.uint8)

def renderPredictions(dataset, predictions):
    # print('predictions are', predictions.values())
    day2pred = {}
    for pt, pred in predictions.items():
        burnName, date, location = pt
        day = (burnName, date)
        if day not in day2pred:
            day2pred[day] = []
        pair = (location, float(pred))
        print('storing prediction', pair)
        day2pred[day].append(pair)

    # print(day2pred)
    results = {}
    for (burnName, date), locsAndPreds in day2pred.items():
        print('locs and preds', locsAndPreds)
        locs, preds = zip(*locsAndPreds)
        # print('reds:', preds)
        xs,ys = zip(*locs)
        preds = [pred+1 for pred in preds]
        print('for burn and date', burnName, date)
        # print((xs,ys))
        print(max(preds), min(preds))
        # print(len(xs), len(preds))
        burn = dataset.data.burns[burnName]
        canvas = np.zeros(burn.layerSize, dtype=np.float32)
        # print(canvas)
        canvas[(xs,ys)] = np.array(preds, dtype=np.float32)
        results[(burnName, date)] = canvas
        plt.figure(burnName +" " + date)
        plt.imshow(canvas/2)
        plt.show()
    return results

def visualizePredictions(dataset, predictions):
    renders = renderPredictions(dataset, predictions)


    # w,h = dataset.data.shape
    # result = np.full((w,h,3), (0,0,255), dtype=np.uint8)
    # endingPerim = dataset.data.output
    # result[endingPerim==1] = (255,0,0)
    # startingPerim = renderLayer(dataset, 'perim', which='all')
    # result[startingPerim==255] = (0,255,0)
    # xs,ys = dataset.indices
    # result[xs,ys,0] = predictions*255
    # result[xs,ys,1] = predictions*255
    # result[xs,ys,2] = predictions*255
    # return result

def normalize(layer):
    '''Rescale to between 0 and 1'''
    l = layer-layer.min()
    m = l.max()
    if m != 0:
        l = l/l.max()
    return l

def show(*imgs, imm=True):
    try:
        for i, img in enumerate(imgs):
            plt.figure(i)
            plt.imshow(img)
        if imm:
            plt.show()
    except:
        print("Not able to show because plt not imported")

def save(img, name):
    fname = 'output/imgs/{}.png'.format(name)
    cv2.imwrite(fname, img)

def saveModel(model):
    from keras.utils import plot_model
    timeString = strftime("%d%b%H:%M", localtime())
    fname = 'output/modelViz/{}.png'.format(timeString)
    plot_model(model, to_file=fname, show_shapes=True)
