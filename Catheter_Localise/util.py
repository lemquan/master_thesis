__author__ = 'quale'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

def plot_filters(path, net, layer, x, y, it):

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
    #fn = '/Users/quale/Dropbox/Project/Code/figs/conv' + str(layer) +'_iter_' \
    #                                            + str(it) + '.png'

    fn = path +'conv' +str(layer) +'_iter_' + str(it) + '.png'
    for j in range(len(filters)):
        ax = fig.add_subplot(y, x, j)
        ax.matshow(filters[j][0], cmap = cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

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


def load_data_for_model(path=None):
    '''
    Call this function to load the data when setting up the neural network/model
    :param path: The location of the catheter_images.npz
    :return:
    '''
    if path == None:
        path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/new_catheter_images_SM.npz'

    f = np.load(path)
    train_input = f['train_X']
    train_label = f['train_y']

    valid_input = f['valid_X']
    valid_label = f['valid_y']

    test_input = f['test_X']
    test_label = f['test_y']

    return train_input, train_label, valid_input, valid_label, test_input, test_label

def plot_images(src_img, src_mask, pred_mask, test_idx, pixels=None, alpha=0.1):
    if pixels == None:
        x = 96
        y = 96
    else:
        x,y = pixels
    src = np.reshape(src_img, (x, y))
    smask = np.reshape(src_mask, (x,y))
    pmask = np.reshape(pred_mask, (x,y))
    sx, sy = np.where(smask == 1.0)

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(src, cmap=cm.Greys_r)
    axarr[0,0].plot(sy,sx, 'rx') # plot the catheter in the orig image
    axarr[0,0].set_title('Original Image')
    axarr[0,0].set_xticks([])
    axarr[0,0].set_yticks([])

    axarr[0,1].imshow(smask, cmap=cm.Greys_r)
    axarr[0,1].set_title('Original Image Binary Mask')
    axarr[0,1].set_xticks([])
    axarr[0,1].set_yticks([])

    axarr[1,0].imshow(pmask, cmap=cm.Greys_r)
    axarr[1,0].set_title('Predicted Binary Mask')
    axarr[1,0].set_xticks([])
    axarr[1,0].set_yticks([])

    px, py = np.where(pmask >= alpha)
    axarr[1,1].imshow(src, cmap=cm.Greys_r)
    axarr[1,1].plot(py,px, 'gx')
    axarr[1,1].set_title('Predicted Catheter Location')
    axarr[1,1].set_xticks([])
    axarr[1,1].set_yticks([])

    # turn off ticks # TODO: FIX THIS
    #plt.savefig('/home/quanle/Project_2015/Catheter_Localise/figs/test_output_'+ str(test_idx) + '.png')
    plt.savefig('/Users/quale/Dropbox/Project/Code/Catheter_Localise/figs/test_output_'+ str(test_idx) + '.png')
    plt.close()


def simple_plot(x, pixel=None):
    if pixel == None:
        x1 = 96
        x2 = 96
    else:
        x1 = pixel[0]
        x2 = pixel[1]
    plt.imshow(np.reshape(x, (x1,x2)), cmap=cm.Greys_r)
    plt.show()


def plot_boxes(input, target, pred, prob, idx, save_path, pixels=(96,96)):
    x,y = pixels
    src = np.reshape(input, (x, y))
    truth = np.reshape(target, (x,y))
    pred = np.reshape(pred, (x,y))

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(src, cmap=cm.Greys_r)
    axarr[0,0].set_title('Input Image')
    axarr[0,0].set_xticks([])
    axarr[0,0].set_yticks([])

    axarr[0,1].imshow(truth, cmap=cm.Greys_r)
    axarr[0,1].set_title('Target Image')
    axarr[0,1].set_xticks([])
    axarr[0,1].set_yticks([])

    axarr[1,0].imshow(pred, cmap=cm.Greys_r)
    axarr[1,0].set_title('Predicted Output')
    axarr[1,0].set_xticks([])
    axarr[1,0].set_yticks([])

    pdf = axarr[1,1].imshow(prob, cmap=cm.hot)
    axarr[1,1].set_title('Probability Density of Prediction')
    axarr[1,1].set_xticks([])
    axarr[1,1].set_yticks([])
    cbar = f.colorbar(pdf, ticks=[], orientation='vertical')

    #plt.savefig('/Users/quale/Desktop/white_box_res/test_output_'+ str(idx) + '.png')
    #plt.savefig('/Users/quale/Desktop/mixed_rects_res/test_output_'+ str(idx) + '.png')
    #plt.savefig('/Users/quale/Desktop/caths_no_bg_res/test_output_'+ str(idx) + '.png')
    #plt.savefig('/Users/quale/Desktop/caths_no_bg_thick_res/test_output_'+ str(idx) + '.png')
    plt.savefig(save_path + '/test_output_'+ str(idx) + '.png')

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size)
        ax.add_patch(rect)
    ax.autoscale_view()
    ax.invert_yaxis()
    plt.show()

# http://nbviewer.ipython.org/gist/felixlaumon/409692acdecc0921eca6
# https://github.com/Lasagne/Lasagne/issues/106
def plot_conv_weights(weights,it):
    fx = weights.shape[0]
    weights = weights[100:]
    gs = gridspec.GridSpec(10,10)
    for i in range(100):
        print 'working on ...', i
        wmin = float(weights[i].min())
        wmax = float(weights[i].max())
        weights[i] *= (255.0/float(wmax-wmin))
        weights[i] += abs(wmin)*(255.0/float(wmax-wmin))

        g = gs[i] # set up the grid
        ax = plt.subplot(g)
        ax.grid()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(weights[i], cmap=cm.Greys_r)
    #plt.show()
    plt.savefig('conv_layer_' + str(it) + '.png')


def plot_cost(path, epochs):
    f = np.load(path)
    train_history = f['train_err']
    valid_history = f['valid_err']


    # th = []
    # tbatch = len(train_history)/epochs
    # for i in range(epochs):
    #     t = np.mean(train_history[i*tbatch:(i+1)*tbatch])
    #     th.append(t)
    #
    # vh = []
    # vbatch = len(valid_history)/epochs
    # for k in range(epochs):
    #     v = np.mean(valid_history[k*vbatch:(k+1)*vbatch])
    #     vh.append(v)

    plt.plot(train_history, '-b', label='Training')
    plt.plot(valid_history, '-r', label='Validation')
    #plt.ylim([84, 89])
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.title('Training and Validation Error on Real Catheters and Wider Synthetic Catheters')
    plt.show()



if __name__ == '__main__':
    path='/home/quanle/Project_2015/Catheter_Localise/new_catheter_images_BIG.npz'
    #train_input, train_label, valid_input, valid_label, test_input, test_label \
    #                                            = load_data_for_model(path)

    #print train_input.shape, valid_input.shape, test_input.shape
    #hinton(np.random.rand(20, 20) - 0.5)

    # plotting the cost function s
    path = '/Users/quale/Desktop/mix/train_valid_caths_thicks_history.npz'
    plot_cost(path, 95)




