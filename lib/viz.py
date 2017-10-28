import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
    PLT = True
except:
    PLT = False

def reassemblePredictions(predictions, indices, shape):
    assert type(shape) == type((1,))
    assert len(shape) >= 2
    result = np.zeros(shape[:2], dtype=np.uint8)
    # remap from 0 and 1 to 127 and 255
    predictions = predictions*128+127
    result[indices] = predictions
    return result

if PLT:
    def show(*imgs):
       for i, img in enumerate(imgs):
           plt.figure(i)
           plt.imshow(img)
       plt.show()
