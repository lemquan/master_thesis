"""
Perform catheter detection and location using Lasange and Theano. Lasagne is the framework to run 
CNNs. 
"""

import time
from util import load_data_for_model, simple_plot, data_preprocess, plot_boxes, plot_conv_weights
import numpy as np
import theano
import theano.tensor as T
import lasagne
from sklearn.metrics import mean_squared_error

def load_data(path, pixel):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    print '... loading data'
    train_input, train_label, valid_input, valid_label, test_input, test_label \
                                                    = load_data_for_model(path)

    # scale the  data
    train_input = train_input/255.0
    valid_input = valid_input/255.0
    test_input = test_input/255.0

    # zero center the data
    train_input, mu = data_preprocess(train_input, mean=None)
    valid_input, _ = data_preprocess(valid_input, mu)
    test_input, _ = data_preprocess(test_input, mu)

    # normalise the data by divding the standard deviation
    std = np.std(train_input)
    train_input /= std
    valid_input /=std
    test_input /= std

    train_input = np.reshape(train_input, (-1, 1, pixel[0], pixel[1]))
    valid_input = np.reshape(valid_input, (-1, 1, pixel[0], pixel[1]))
    test_input = np.reshape(test_input, (-1, 1, pixel[0], pixel[1]))

    return train_input, train_label, valid_input, valid_label,\
            test_input, test_label
    # multiply labels by 255.0 if noisy_box dataset or caths_96 cus the values are 0 or 1. want 0 or 255- regression
    # divide labels by 255.0 if classification

# ##################### Build the neural network model #######################
def build_cnn(input_size, output_size, input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, input_size, input_size),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(11, 11), # 32, 11x11
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
             network, num_filters=256, filter_size=(2, 2),
             nonlinearity=lasagne.nonlinearities.rectify)

    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.3),
            num_units=1000,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.2),
            num_units=output_size,
            nonlinearity=lasagne.nonlinearities.linear, name='output_layer') # change to sigmoid for classification or linear for regression

    return network


# ############################# Batch iterator ###############################
def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(X_train, y_train, X_val, y_val, X_test, y_test, num_epochs=150, batchsize=200):
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    input_size = pixel[0]
    output_size = pixel[0] * pixel[1]
    network = build_cnn(input_size,output_size, input_var)

    #L2 Regularisation
    l2_reg = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var) #change to binary for classifcation, squared errror for reg
    loss = loss.mean() + 0.001*l2_reg
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0001, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy for CLASSIFICATION
    #test_prediction = T.argmax(test_prediction, axis=1)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_prediction])

    print("Starting training...")
    # We iterate over epochs:
    train_hist = []
    valid_hist = []
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize):
            inputs, targets = batch
            err = train_fn(inputs, targets)
            train_err += err
            #train_hist.append(err)
            train_batches += 1
        get_convo_filters(network, epoch) # get filters

        # curr_valid = 0
        # best_valid = np.inf
        # best_valid_epoch = 0
        # patience = 200
        # run on the validation set
        val_err =0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize):
            inputs, targets = batch
            valid_err, valid_pred = val_fn(inputs, targets)
            val_err+= valid_err
            val_batches+=1
        # early stopping to prevent overfitting
        curr_valid = val_err
        if curr_valid < np.inf:
            best_valid = curr_valid
            best_valid_epoch = epoch
            best_weights = lasagne.layers.get_all_param_values()
        elif best_valid_epoch+patience < epoch:
            print('Early stopping!')

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

        train_hist.append((train_err / train_batches))
        valid_hist.append((val_err / val_batches)) # save the history

    # save the training and valid erro #TODO: change
    np.savez('train_valid_caths_thicks_history', train_err=np.asarray(train_hist), valid_err=np.asarray(valid_hist))
    ############################# After training,compute and print the test error ###########################
    #lasagne.layers.set_all_param_values(network, best_weights) # reload the best params

    test_err = 0
    test_batches = 0
    preds = []
    for batch in iterate_minibatches(X_test, y_test, batchsize):
        inputs, targets = batch
        err, test_pred = val_fn(inputs, targets)
        preds.append(test_pred)
        test_err += err
        test_batches += 1
    np.savez('preds_train_valid_caths_thicks_history.npz', preds) #TODO: change
    print 'final test err', test_err

################################### Helper functions #########################################
def get_convo_filters(network, epoch):
    print 'saving convolutional filters at epoch:', epoch
    layers = lasagne.layers.get_all_layers(network) # get all the layers
    layerCnt = 0
    for l in layers:
        if type(l) == lasagne.layers.Conv2DLayer: # find a convolution layer
            weights = l.W.get_value()
            weights = weights.reshape(weights.shape[0]*weights.shape[1], weights.shape[2], weights.shape[3])
            for i in range(weights.shape[0]):
                wmin = float(weights[i].min())
                wmax = float(weights[i].max())
                weights[i] *= (255.0/float(wmax-wmin))
                weights[i] += abs(wmin)*(255.0/float(wmax-wmin))
            f = open('/home/quanle/Project_2015/Boxes/filters/conv_layer_'+ str(layerCnt)+ '_'+ str(epoch) + '.weights', 'wb')
            np.save(f, weights)
            layerCnt+=1 # increase layer count

def plot_conv_filters():
    path = '/Users/quale/Desktop/mix/filters/conv_layer_3_50.weights'
    #path = '/Users/quale/Desktop/filters/conv_layer_0_16.weights'
    with open(path, 'rb') as f:
        layer = np.load(f)
        plot_conv_weights(layer,1)


def check_preds(X_test, y_test):
    #path = '/Users/quale/Desktop/caths_preds_96_reg/preds_caths_96.npz'
    #path = '/Users/quale/Desktop/white_boxes_res/preds_white_boxes.npz'
    #path='/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/preds_rects_big_reg.npz'
    path='/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/preds_mixed_caths_thick.npz'
    save_path='/Users/quale/Desktop/mix'
    f = np.load(path)
    f_out = f['arr_0']
    preds = np.reshape(f_out, (f_out.shape[0]*f_out.shape[1], 9216))
    total = preds.shape[0]

    #y_test = y_test*255.0 # do this for caths_96 and noisy_box data sets
    se = np.sum((preds - y_test[0:total])**2, axis=1) / 9216.0
    mse = np.sum(se) / total * 1.0
    rmse = mean_squared_error(y_test[0:total], preds)
    print 'MSE:', mse, rmse

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    all = 0
    total_gt =0
    for i in range(preds.shape[0]): #preds.shape[0]
        print 'working on', i
        # get indexes of ground truth binary mask
        p = np.copy(preds[i]) #1

        max_intensity = max(p) / 2
        p[p >= max_intensity] = 255.0
        p[p < max_intensity] = 0.0

        gt_pos_idx = np.where(y_test[i].flatten() == 255.0)
        gt_neg_idx = np.where(y_test[i].flatten() == 0.0)

        preds_pos_idx = np.where(p== 255.0)
        preds_neg_idx = np.where(p== 0.0)
        #print len(gt_pos_idx[0]), len(preds_pos_idx[0])
        #print len(gt_neg_idx[0]), len(preds_neg_idx[0])
   #     total_gt += len(gt_pos_idx[0])
    #    tp += (len([w for w in preds_pos_idx[0] if w in gt_pos_idx[0]])) # true positives
    #     fp += (len([w for w in preds_pos_idx[0] if w not in gt_pos_idx[0]])) #false positives
    #
    #     tn += (len([w for w in preds_neg_idx[0] if w in gt_neg_idx[0]])) # true negatives
    #     fn += (len([w for w in preds_neg_idx[0] if w not in gt_neg_idx[0]])) #false negatives
    #     # plot the prediction
        yhat = np.reshape(preds[i], (96,96))
        plot_boxes(X_test[i], y_test[i],p, yhat, i, save_path)
    # if tp+fp != 0:
    #     precision = tp*1.0 / (tp + fp)
    # else:
    #     precision = -9999999
    #
    # if tp + fn !=0:
    #     recall =  tp*1.0 / (tp + fn)
    # else:
    #     recall = -9999999
    # print 'tp:', tp, 'fp:', fp, 'tn:', tn, 'fn:', fn
    # print precision, recall
    # print 'f1 score:', ((1.0*precision*recall) / (precision + recall)) * 2.0

    print tp / (total_gt*1.0)


if __name__ == '__main__':
    epochs = 95
    batch_size = 150

    print("Loading data...")
    pixel = (96,96)
    #path = '/home/quanle/Project_2015/Boxes/white_boxes_ds.npz'
    #path = '/home/quanle/Project_2015/Boxes/noisy_boxes_ds.npz'
    #path = '/home/quanle/Project_2015/Boxes/rectangles_ds.npz'
    #path = '/home/quanle/Project_2015/Boxes/rectangles_big_ds.npz'
    #path = '/home/quanle/Project_2015/Boxes/mixed_rects_ds.npz'
    #path = '/home/quanle/Project_2015/Boxes/rects_5_ds.npz'
    #path = '/home/quanle/Project_2015/Boxes/rects_thin_1_ds.npz'
    #path = '/home/quanle/Project_2015/Boxes/caths_no_bg_mixed_rects_ds.npz'
    #path = '/home/quanle/Project_2015/Boxes/caths_no_bg_thick_mixed_rects_ds.npz'
    #path = '/home/quanle/Project_2015/Boxes/caths_org_and_thicks_mixed_ds.npz'
    #path = '/home/quanle/Project_2015/Catheter_Localise/catheter_images_96.npz'

    #path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/white_boxes_ds.npz'
    #path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/noisy_boxes_ds.npz'
    #path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/rectangles_big_ds.npz'
    #path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/mixed_rects_ds.npz'
    #path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/rects_5_ds.npz'
    #path='/Users/quale/Dropbox/Project/Code/Catheter_Localise/catheter_images_96.npz'
    #path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/rects_thin_1_ds.npz'
    #path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/caths_no_bg_mixed_rects_ds.npz'
    #path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/caths_no_bg_thick_mixed_rects_ds.npz'
    path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/testing_boxes/caths_org_and_thicks_mixed_ds.npz'

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(path, pixel)
    main(X_train, y_train, X_val, y_val, X_test, y_test, epochs, batch_size)
    check_preds(X_test, y_test)