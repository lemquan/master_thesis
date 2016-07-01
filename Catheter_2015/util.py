__author__ = 'quale'

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py

def plot_filters(net, layer, x, y, it):

    """Plot the filters for net after the (convolutional) layer number
    layer.  They are plotted in x by y format.  So, for example, if we
    have 20 filters after layer 0, then we can call show_filters(net, 0, 5, 4) to
    get a 5 by 4 plot of all filters.

    it- epoch or iternation number to name plot when saving

    """
    filters = net.layers[layer].W.get_value(borrow=True)
    fig = plt.figure()
    t = 'Weights of Convolution Layer' + str(layer) + 'at iter: ' + str(it)
    fig.suptitle(t)
    fn = '/Users/quale/Dropbox/Project/Code/figs/conv' + str(layer) +'_iter_' \
                                                 + str(it) + '.png'

    #fn = '/home/quanle/Project_2015/figs/conv' + str(layer) +'_iter_' \
    #                                             + str(it) + '.png'
    for j in range(len(filters)):
        ax = fig.add_subplot(y, x, j)
        ax.matshow(filters[j][0], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

def load_data_for_model(path=None):
    '''
    Call this function to load the data when setting up the neural network/model
    :param path: The location of the catheter_images.npz
    :return:
    '''
    if path == None:
        path = '/Users/quale/Dropbox/Project/Code/Catheter_2015/catheter_images_BIG.npz'

    f = np.load(path)
    train_input = f['train_input']
    train_label = f['train_label']

    valid_input = f['valid_input']
    valid_label = f['valid_label']

    test_input = f['test_input']
    test_label = f['test_label']

    return train_input, train_label, valid_input, valid_label, test_input, test_label


if __name__ == '__main__':
    #path = '/home/quanle/Project_2015/catheter_images_BIG.h5'
    train_input, train_label, valid_input, valid_label, test_input, test_label \
                                                = load_data_for_model()

    print train_input.shape, valid_input.shape, test_input.shape



