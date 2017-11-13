from time import localtime, strftime
import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
except:
    pass
from keras.utils import plot_model

def reassemblePredictions(predictions, indices, shape):
    assert type(shape) == type((1,))
    assert len(shape) >= 2
    result = np.zeros(shape[:2], dtype=np.uint8)
    # remap from 0 and 1 to 127 and 255
    predictions = predictions*128+127
    result[indices] = predictions
    return result

# def show(*imgs):
#     for i, img in enumerate(imgs):
#         plt.figure(i)
#         plt.imshow(img)
#     plt.show()

def saveModel(model):
    timeString = strftime("%d%b%H:%M", localtime())
    fname = 'output/modelViz/{}.png'.format(timeString)
    plot_model(model, to_file=fname, show_shapes=True)
