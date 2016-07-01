import csv
import os
import cv2
import h5py
import theano
import numpy as np

'''
    This scripts load the first few folders of fluoroscopic images from CatheterLabels.

    Centering data will result in negative values:
    http://stackoverflow.com/questions/29743523/subtract-mean-from-image
'''

def read_csv(path, gt):
    '''
    Reads in the csv file that contains the coordinates of each catheter. Each folder has its own label.csv
    :param path: the absolute path of the folder
    :param gt: labels.csv file
    :return: csv file in list structure
    '''
    res = []
    # get the labels.csv/gt
    with open(path + gt, 'rb') as file:
        labels = csv.reader(file, delimiter=',')
        for row in labels:
            res.append(row)
    return res

def load_images(path, pixel_res):
    '''
    Function load the catheter and non catheter images. Also convert each image into grayscale
    :param path:
    :return:
    '''
    ret = 0
    folders = []
    pos_images = []
    ########################### load the positive images ###########################
    for filename in os.listdir(path):
        folders.append(filename) # get all the folders of catheter images
    folders.pop(0) # remove the first entry which .DS file

    # loop through each folder containing the images
    for i in range(len(folders)):
        folder_path = path + '/'+ folders[i]
        print 'working on...', folder_path
        image_folders = os.listdir(folder_path)
        image_folders.pop(0) # remove the private file

        # load each images
        for img_name in image_folders:
            if 'label' not in img_name:
                img = cv2.imread(folder_path+'/'+img_name)
                img = cv2.resize(img, pixel_res) # resize the image for training

                # convert to grayscale
                gray_pos = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                tup = (gray_pos, 1) # 1 for catheter image, positive
                pos_images.append(tup)
    print 'size of positive images:', len(pos_images)
    ######################### load the negative images ###########################
    print 'working on negative images.....'
    neg_images = []
    neg_image_folder = '/Users/quale/Desktop/Neg_images'
    neg_images_path = os.listdir(neg_image_folder)
    neg_images_path.pop(0) # remove the .DS file
    for path in neg_images_path:
        nimg = cv2.imread(neg_image_folder+'/'+path)
        nimg = cv2.resize(nimg, pixel_res)
        gray_neg = cv2.cvtColor(nimg, cv2.COLOR_RGB2GRAY)
        tup = (gray_neg, 0) # 0 = negative image, not catheter
        neg_images.append(tup)
    print 'size of negative images:', len(neg_images)
    ################### combine the images ######################
    all_images = np.concatenate((pos_images, neg_images), axis=0)
    train_input, train_label, valid_input, valid_label, test_input, test_label = \
                                                    split_data(all_images)
    ################# pre-process the data #####################
    ctrain, mu = data_preprocess(train_input) # center the data
    cvalid, _ = data_preprocess(valid_input, mu)
    ctest, _ = data_preprocess(test_input, mu)

    ################# write to numpy file #########################
    print 'Writing to file.....'
    fn = 'catheter_images_detect'
    ctrain = np.asarray(ctrain, dtype=theano.config.floatX)
    train_label = np.asarray(train_label, dtype=theano.config.floatX)
    cvalid = np.asarray(cvalid, dtype=theano.config.floatX)
    valid_label = np.asarray(valid_label, dtype=theano.config.floatX)
    ctest = np.asarray(ctest, dtype=theano.config.floatX)
    test_label = np.asarray(test_label, dtype=theano.config.floatX)
    np.savez(fn, train_input=ctrain, train_label=train_label, valid_input=cvalid, \
                                valid_label=valid_label, test_input=ctest, test_label=test_label)
    ret = 1
    return ret



def split_data(dataset):
    '''
    Split the dataset into training, validation, and test set. The split will be 70, 20, 10
    :param dataset: the entire dataset containing positive and negative images. each row is a tuple containing the data and label
    :param pixel_res: the number of neurons per image
    :return: a tuple of training, validation, and test
    '''

    # shuffle the data to randomize
    shuffled = np.random.permutation(dataset)

    # get the sizes based on split percentage for each set
    print 'total dataset size:', len(shuffled)
    ts = np.trunc(len(shuffled) * 0.64) # train
    print ts
    vs = np.trunc(len(shuffled) * 0.16)# valid
    print vs
    tts = np.trunc(len(shuffled) * 0.20 )# test
    print tts

    train_input, train_label = get_input_label(shuffled[:ts])
    valid_input, valid_label = get_input_label(shuffled[ts:ts+vs])
    test_input, test_label = get_input_label(shuffled[ts+vs:])

    return train_input, train_label, valid_input, valid_label, test_input, test_label

def get_input_label(data):
    '''
    This function separates the grouped data from its input and the labels
    :param data:
    :return:
    '''
    x_set, y_set = zip(*data) # separate input from its label
    input = [x.flatten() for x in x_set]
    label = [y for y  in y_set]
    return input, label

def vectorized_result(j):
    """Return a 2-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. [1,0] = negative image, [0, 1] = positive image
    """
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

def data_preprocess(input, mean=None):
    '''
    Function to pre process the images.
    An important point to make about the preprocessing is that any preprocessing statistics (e.g. the data mean)
    must only be computed on the training data, and then applied to the validation / test data.
    :param image: a grey scale 1D of the image
    :return: mean subtracted and normalized image..still 1D
    '''

    if mean == None:
        mean = np.mean(input)
    # mean subtraction to center the data to 0
    center = input - mean
    return center, mean

def main():
    '''
    Run this main to load positive and negative images. And perform a preprocess.
    After main is done, the complete dataset to train the model should be ready: train_input, train_label,
    valid_input, valid_label, test_input, and test_label. This will create an h5 file that needs to be loaded in order
    to train the CNN.
    :return:
    '''
    path = '/Users/quale/Desktop/CatheterLabels'
    res = load_images(path, (96,96)) # recommended pixel size, used commonly in ImageNet challenges (224, 224)

    #path = '/Users/quale/Dropbox/Project/Code/Numpy_Load/catheter_images_BIG.npz'
    # ims = np.load(path)
    # print ims.files
    # ims['arr_0']



def paint(img):
    cv2.imshow('new image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    theano.config.floatX = 'float32'
    main()