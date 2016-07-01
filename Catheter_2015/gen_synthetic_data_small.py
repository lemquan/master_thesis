__author__ = 'quale'
'''
    This script generates synthetic data for wires on fluoroscopic images. This is used independently.
'''

import numpy as np
import cv2

from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
import h5py



def gen_synthetic_data():
    '''
        A function to synethetic generate coordinates of a wire using spline fit
    :return: A tuple where index = 0 is the x coordinates and index=1 is y coordinates of the spline
    '''
    syn_data_x = np.zeros((5000,300))
    syn_data_y = np.zeros((5000,300))
    for i in xrange(5000):
        # randomly select how many points for hte cather body and tip
        init_x = np.random.randint(5,10)
        x = np.linspace(-init_x, init_x,300)

        # this equation was chosen for it resembled the most as a wire
        y = np.sin((x**3/6) * 0.05+np.random.rand()) + (0.5*np.random.rand(len(x)))

        # randomly select the smoothness (how many knots)
        s = np.random.randint(120,500)

        # randomly select the degree of smoothing
        k = np.random.randint(2,5)
        spl = UnivariateSpline(x,y, k=k, s=s)
        spl_y = spl(x)

        syn_data_x[i,:] = x
        syn_data_y[i,:] = spl_y
        # plt.plot(x,y, 'b-', ms=5)
        # plt.plot(x, spl_y, 'g', lw=4)
        # plt.show()
    with h5py.File('synthetic_data.h5', 'w') as hf:
        hf.create_dataset('syn_data_x', data=syn_data_x)
        hf.create_dataset('syn_data_y', data=syn_data_y)
    return 'synthetic_data.h5'


def blend_to_background(path):

    with h5py.File(path,'r') as hf:
        # Print a list of arrays in this file
        keys = hf.keys()

        # this is the synethetic generated coordinates of a guide wire (x,y)
        syn_data_x = np.array(hf.get(keys[0]))
        syn_data_y = np.array(hf.get(keys[1]))

        for i in xrange(len(syn_data_x)):
            bg = cv2.imread('background.png', 0) #load the background in greyscale
            # cv2.imshow('org background', bg)
            # cv2.waitKey(0)

            # scale the coordinates for the guide wires, and as int
            scaled_syn_x = scale_syn_data(syn_data_x[i,:], np.ndarray.min(syn_data_x), np.ndarray.max(syn_data_x))
            scaled_syn_y = scale_syn_data(syn_data_y[i,:], np.ndarray.min(syn_data_y), np.ndarray.max(syn_data_y))
            fin_image = interpolate_syn_data(scaled_syn_x, scaled_syn_y, bg)

            fn = 'syn_images/' + 'syn_image_'+ str(i) +'.png'
            cv2.imwrite(fn, fin_image)
    #print 'done generating synthetic images'


def interpolate_syn_data(x,y, bg):
    '''
    zero is taken to be black, and 255 is taken to be white. Values in between make up the different shades of gray
    :param x:
    :param y:
    :param bg: background image, size is 512x512, dtype=uint8
    :return: background with synthetic guide wire, dtype=uint8
    '''

    # a mix of operations on the background image including flipping, rotations, and contrast
    bg = interpolate_bg(bg)

    # create a blank canvas for the background and wire to be placed on
    canvas = 255*np.ones((512,512), dtype=np.uint8)

    # thicken the wire by padding the x-coordinates to the left and right with the same value
    width = 2
    for i in xrange(len(x)):
        x_coord = x[i].astype(np.int32)
        y_coord = y[i].astype(np.int32)
        canvas[y_coord, x_coord-width:x_coord+width+1] = 0

    # rotate the guide wire
    rows, cols = bg.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
    r = cv2.warpAffine(canvas, M, (cols, rows))

    # apply a Gaussian blur to smooth the wire
    blur = cv2.GaussianBlur(r,(3,9), 0)

    # darken the wire by changing contrast
    darken = contrast_wire(blur, 1, 1)

    # blend two images
    blend = cv2.addWeighted(bg,1,darken, 0.2, 1)

    # debugging images
    # r = np.hstack((bg, blend))
    # cv2.imshow('syn data', r)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return blend

def scale_syn_data(x, old_min, old_max):
    '''
    Function to scale the synthetically generated coordinates to a given background image
    http://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
    :param x:
    :param old_min:
    :param old_max:
    :return:
    '''
    new_min = 0
    new_max = 510 # raw image is 512, but took away 2 pixels to compensate for future transformations

    new_value = (((x - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    new_value = np.floor(new_value)
    return new_value.astype(int)

def contrast_wire(image, theta, phi):
    '''
    :param image:
    :param theta: if bg, keep as 1. for guide wire, keep theta and phi as 1
    :param phi: if bg, in range 1-1.9
    :return:
    '''
    # Decrease intensity such that
    # dark pixels become much darker,
    # bright pixels become slightly dark

    max_intensity = 255.0
    # theta = 1
    # phi = 1.89
    darken = (max_intensity/phi)*(image/(max_intensity/theta))**2
    darken = np.array(darken,dtype=np.uint8)
    return darken

def interpolate_bg(img):
     bg = img

     op = np.random.randint(0,2)
     if op: # flip the image
        bg = cv2.flip(bg,np.random.randint(-1,2))

     op = np.random.randint(0,2)
     if op: # perform a rotation
         rows, cols = bg.shape
         degree = [90,180,360]
         d = np.random.randint(0,3)
         M = cv2.getRotationMatrix2D((cols/2, rows/2), degree[d], 1)
         bg = cv2.warpAffine(bg, M, (cols, rows))

     op = np.random.randint(0,4)
     if op == 0: # lighten with equalizer
        bg = cv2.equalizeHist(bg)
     elif op == 1: # ligten with CLAHE
         clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(20,20))
         bg = clahe.apply(bg)
     elif op ==2: # darken the image
         bg = contrast_wire(bg, 0.8, 1.02)
     else: # do nothing
         bg = bg

     #cv2.imshow('new iamge',res)
     # cv2.imshow('clahe', bg)
     # cv2.waitKey(0)
     # cv2.destroyAllWindows()
     return bg

def create_neg_images(img_path):
    #facehttps://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations
    img = cv2.imread(img_path,0)
    # plt.imshow(img),plt.title('Input')
    # plt.show()

    # points for endo.jpg
    pts1 = np.float32([[151,2],[485,60],[73,523],[502,504]])
    # pts1 = np.float32([[10,65],[368,52],[9,587],[189,560]])
    pts2 = np.float32([[10,10],[512,10],[10,512],[512,512]])

    # points for foramen.jpg
    #pts1 = np.float32([[136,211],[979,84],[299, 1214],[998,1197]])
    #pts2 = np.float32([[0,0],[512,0],[0,512],[512,512]])
    #M = cv2.getPerspectiveTransform(pts1,pts2)

    #n_img = cv2.warpPerspective(img,M,(512,512))

    for i in xrange(2001):
        fn_image = interpolate_bg(img)

        fn = 'neg_images/' + 'neg_1_image2_'+ str(i) +'.png'
        cv2.imwrite(fn, fn_image)
    # cv2.imshow('new', n_img )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def exp_transfer(img):
    '''
    Using exponential transfer to change opacity of the synethic wire. Not used because of the array type. Values are
    very tiny, requires float data type, but pictures are in uint8. Canvas is now uint8 so func no longer works
    :param img:
    :return:
    '''
    c0 = [0.08, 0.12]
    c1 = [2200, 3800]

    #idx = np.random.randint(0,2, 2)
    idx = [1, 0] #personal choice
    alpha =  ( c0[idx[0]] * (np.exp(img/c1[idx[0 ]]) - 1) ) / (np.exp(7500.0/c1[idx[0]]) - 1)
    img = alpha
    return img


def main():

    # r = cv2.imread('/Users/quale/Desktop/CatheterWarps/TAVI_PER1F/frame_00000.png')
    # gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('test',gray)
    # cv2.waitKey(0)

    # blend the synethetic wire to the background, perform some contrasts to make images more realistic and varies
    #gen_synthetic_data()
    #blend_to_background('synthetic_data.h5')


    # generate negative images
    #img_path='/Users/quale/Desktop/endo.jpg'
    #img_path='/Users/quale/Desktop/foramen.jpg'
    img_path = '/Users/quale/Desktop/neg_images/new_neg_1.png'
    create_neg_images(img_path)


if __name__ == '__main__':
    main()





