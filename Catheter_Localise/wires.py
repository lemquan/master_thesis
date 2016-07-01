__author__ = 'quale'

'''
    This script uses the wires that are synethetically generated and from the original data source. We use this to create
    a new data set containing the catheters, but without any background images that acts as noise.
'''

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np

path = '/Users/quale/Desktop/CatheterLabels'
spath = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/org_caths/'
folders = []
for f in os.listdir(path):
    folders.append(f)
folders.pop(0)

for i in range(1):
    folder_path = path + '/'+folders[i]
    images = os.listdir(folder_path)
    images.pop(0)

    ct = 0
    for img in images:
        if 'label_' in img:
            mask = cv2.imread(folder_path+'/'+img)
            mask = cv2.resize(mask, (96,96))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            org = 255.0*np.ones((96,96)) - mask

            org_fn = spath + 'cath_' + str(ct) + '.png'
            mask_fn = spath + 'mask_' + str(ct) + '.png'
            plt.imsave(org_fn, org, cmap=cm.Greys_r)
            plt.imsave(mask_fn, mask, cmap=cm.Greys_r)
            ct+=1


