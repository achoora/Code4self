import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
inputImage = cv2.imread("index.jpeg")
inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(inputImageGray,150,200,apertureSize = 3)
h, w = edges.shape[:2]
filled_from_bottom = np.zeros((h, w))
for col in range(w):
    for row in reversed(range(h)):
        if edges[row][col] < 255: filled_from_bottom[row][col] = 255
        else: break

minLineLength = 30
maxLineGap = 5
lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
        pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
        cv2.polylines(inputImage, [pts], True, (0,255,0))

font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(inputImage,"Tracks Detected", (500, 250), font, 0.5, 255)




#edges=cv2.bitwise_not(edges)
fd, hog_image = hog(edges, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
fig = plt.figure(0, figsize = (8,6))
ax1=fig.add_subplot(2,2,1)

ax1.axis('off')
ax1.imshow(edges, cmap=plt.cm.gray)
ax1.set_title('edge highlighted')
# Rescale histogram for better display
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax2=fig.add_subplot(2,2,2)
ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('HOG')
ax3=fig.add_subplot(2,2,3)
ax3.axis('off')
ax3.imshow(inputImage)
ax3.set_title('original')
plt.show()
#cv2.imshow('Detected Edges', inputImage)
#cv2.waitKey(0)
