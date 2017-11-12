import sys
import cv2
import matplotlib.pyplot as plt
fname = sys.argv[1]
img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
plt.imshow(img)
plt.show()
