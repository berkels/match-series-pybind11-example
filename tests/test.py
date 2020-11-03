#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cmake_example
from scipy.ndimage import map_coordinates
import matplotlib as mpl
import matplotlib.pyplot as plt

a = np.array([[0.1, 10, 30], [15, 2, 13]])
cmake_example.array_test(a)
print(a)
print(np.sqrt(np.sum(a**2)))

reference_image = np.zeros((64, 64))
template_image = np.zeros((64, 64))
displacement = np.zeros((2, 64, 64))

reference_image[10:30, 15:45] = 1
template_image[25:45, 25:55] = 1

out = cmake_example.register_images(reference_image, template_image, displacement)

mpl.rcParams["image.cmap"] = 'gray'
_, ax = plt.subplots(nrows=2, ncols=2)
ax[0, 0].title.set_text('reference')
ax[0, 0].imshow(reference_image)
ax[0, 1].title.set_text('template')
ax[0, 1].imshow(template_image)
ax[1, 0].title.set_text('deformed template scipy')
# Convert the displacement to a coordinate system
coords = np.mgrid[0:template_image.shape[0], 0:template_image.shape[1]] \
         + np.multiply([displacement[1, ...], displacement[0, ...]], (np.max(template_image.shape)-1))
ax[1, 0].imshow(map_coordinates(template_image, coords, order=1, mode='nearest', cval=0.0))
ax[1, 1].title.set_text('deformed template returned')
ax[1, 1].imshow((out))
plt.show()
