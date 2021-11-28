# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 07:09:24 2020

@author: abc
"""


import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
model = load_model('model_v_5.h5')
path = input("Enter the image path: ")
img1 = cv2.imread(path)
img1 = cv2.resize(img1, (0, 0), fx = 0.1, fy = 0.1)
cv2.imshow('image', img1)
#path = 'data/test/NORMAL/NORMAL2-IM-0112-0001 - Copy (2).jpeg'
img = image.load_img(path, target_size=(200, 200))
x = image.img_to_array(img)
plt.imshow(x/255.)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes =model.predict(images, batch_size=10)
#print(classes[0])
if classes[0]<0.5:
    print("Patient is normal")
else:
    print("Patient may have covid")

cv2.waitKey(0)
cv2.destroyAllWindows()