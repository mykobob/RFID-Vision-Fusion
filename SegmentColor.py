
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from skimage import feature
from skimage.feature import blob_dog, blob_log, blob_doh
from collections import Counter, defaultdict
import os
import time


# In[2]:


# import the necessary packages
import numpy as np
import cv2

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

params.filterByColor = False
params.filterByArea = False
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False

# Change thresholds
params.minThreshold = 250
params.maxThreshold = 260
params.thresholdStep = 1

resize_val = 2

# # Filter by Area.
params.filterByArea = True
params.minArea = 5000 // (resize_val ** 2)
params.maxArea = 100000000 // (resize_val ** 2)
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)

def color_segmentation(img, num_clusters, lower_brown, upper_brown, covar_type='full'):
    np.set_printoptions(threshold=np.nan)
    img = cv2.resize(img, (img.shape[1] // resize_val, img.shape[0] // resize_val))

    start = time.time()

    mask = cv2.inRange(img, lower_brown, upper_brown)
    result = cv2.bitwise_and(img, img, mask=mask)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    result[result > 0] = 255

    keypoints = detector.detect(result)

    largest_blob, best_size = None, 0
    for key_point in keypoints:
        if key_point.size > best_size:
            best_size = key_point.size
            largest_blob = key_point
    end = time.time()
    
    print('Blob detection took {:.2f} seconds'.format(end - start))

#     im_with_keypoints = cv2.drawKeypoints(result, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
#     print('edges', edges)
    return key_point
#     return blobs_doh, img


# In[3]:


img = cv2.imread("./data/3d_no_occlusions_twisting_1/run_466.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_brown = (115, 115, 60)
# upper_brown = (180, 180, 140)
lower_brown = (0.09 * 255, 0.1 * 255, 100) 
upper_brown = (0.2 * 255, 0.4 * 255, 1* 255) 
biggest_blob = color_segmentation(img, 10, lower_brown, upper_brown, 'full')
# plt.imshow(clustered_img)
# fig, axes = plt.subplots()

# axes.imshow(new_img, interpolation='nearest')
# for blob in blobs_doh:
#     y, x, r = blob
#     c = plt.Circle((x, y), r, color=(0, 0, 1), linewidth=2, fill=False)
#     axes.add_patch(c)
# axes.set_axis_off()


# In[4]:


worldWidth = 273.05
worldHeight = 273.05
FOVx = 50 / 180. * np.pi # left to right
FOVy = 30 / 180. * np.pi 
imwidth = 1280 / 2
imheight = 960 / 2
dbc = DistanceBearingCalculator(worldWidth, worldHeight, FOVx, FOVy, imwidth, imheight)
dbc.getWorldCoordinates(biggest_blob.size, biggest_blob.size, biggest_blob.pt[0], biggest_blob.pt[1])


# In[ ]:


import colorsys
# fake = np.array([[[154, 147, 104, 1], [148, 146, 104, 1], [175, 165, 132, 1], [105, 95, 69, 1], [168, 171, 153, 1], [124, 117, 102, 1], [112, 105, 75, 1],[141, 139, 104, 1]]])
fake = np.array([[[154, 147, 104], [148, 146, 104], [175, 165, 132], [105, 95, 69], [168, 171, 153], [124, 117, 102], [112, 105, 75],[141, 139, 104], [140, 147, 60], [49, 52, 38], [189, 179, 144]]])
print(fake.shape)
for r, g, b in fake.squeeze(0):
#     img = cv2.cvtColor(fake, cv2.COLOR_RGB2HSV)
    print(colorsys.rgb_to_hsv(r, g, b))
# print(img)


# center_away, 0 - 114, 109, 93
# center_away, 40 - 134, 130, 108
# center_away, 49 - 138, 136, 111
# 3d_no_twisting, 1 - 146, 145, 131
# 3d_no_twisting, 341 - 169, 152, 120
