import sys
import cv2
from scipy.misc import imread
import matplotlib.pyplot as plt


class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={}'.format(x, y, z)
try:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    root = Tk()
    root.withdraw()
    fname = askopenfilename()
    root.destroy()
except:
    fname = sys.argv[1]

img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
print('img has shape and dtype:', img.shape, img.dtype)

fig, ax = plt.subplots()
im = ax.imshow(img, interpolation='none')
ax.format_coord = Formatter(im)
plt.show()
