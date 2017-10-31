from time import localtime, strftime
import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
    print('Successfully imported pyplot')
except:
    print('Failed to import pyplot ')


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

def visualizePredictions(dataset):
    w,h = dataset.data.shape
    result = np.full((w,h,3), (0,0,255), dtype=np.uint8)
    endingPerim = dataset.data.output
    result[endingPerim==1] = (255,0,0)
    startingPerim = renderLayer(dataset, 'perim', which='all')
    result[startingPerim==255] = (0,255,0)
    xs,ys = dataset.indices
    result[xs,ys,0] = dataset.predictions*255
    result[xs,ys,1] = dataset.predictions*255
    result[xs,ys,2] = dataset.predictions*255
    return result

def normalize(layer):
    '''Rescale to between 0 and 1'''
    l = layer-layer.min()
    m = l.max()
    if m != 0:
        l = l/l.max()
    return l

def show(*imgs):
    try:
        for i, img in enumerate(imgs):
            plt.figure(i)
            plt.imshow(img)
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
