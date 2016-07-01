import time

import numpy as np
import theano
import theano.tensor as T
from util import load_data_for_model, plot_images, data_preprocess, simple_plot
import lasagne

def load_data(path, pixel):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    print '... loading data'
    train_input, train_label, valid_input, valid_label, test_input, test_label \
                                                    = load_data_for_model(path)

    # normalise data
    train_input = train_input/255.0
    valid_input = valid_input/255.0
    test_input = test_input/255.0

    # train_input, mu = data_preprocess(train_input, mean=None)
    # valid_input, _ = data_preprocess(valid_input, mu)
    # test_input, _ = data_preprocess(test_input, mu)

    train_input = np.reshape(train_input, (-1, 1, pixel[0], pixel[1]))
    valid_input = np.reshape(valid_input, (-1, 1, pixel[0], pixel[1]))
    test_input = np.reshape(test_input, (-1, 1, pixel[0], pixel[1]))

    # if model == 'alexnet':
    #     train_label = np.reshape(train_label, (-1, 1, 96, 96))
    #     valid_label= np.reshape(valid_label, (-1, 1, 96, 96))
    #     test_label = np.reshape(test_label, (-1, 1, 96, 96))

    rval = [(train_input, train_label), (valid_input, valid_label),
            (test_input, test_label)]
    return rval


# ##################### Build the neural network model #######################
def build(input_var=None):
    print '.... building LeNet5'
    # The LetNet5 archteicture

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, pixel[0], pixel[1]),
                                        input_var=input_var, name='input_layer')

    # 1st: Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name='conv_1')

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), name='maxpool_1')

    # 2nd: convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify, name='conv_2')

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), name='maxpool_2')

    # 3rd convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify, name='conv_3')
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), name='maxpool_3')

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2000,
            name='hidden_1',
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected hidden layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.3),
            num_units=1000,
            name='hidden2',
            nonlinearity=lasagne.nonlinearities.rectify)

    # The output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.1),
            num_units=pixel[0]*pixel[1],
            name='output_layer',
            nonlinearity=lasagne.nonlinearities.sigmoid) # TODO: change

    return network

def build_imagenet(input_var=None):
    print 'building ImageNet for Transfer Learning'
    # The ImageNet Archtiecture

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, pixel[0], pixel[1]),
                                        input_var=input_var, name='input_layer')

    # 1st: Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=96, filter_size=(11, 11),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name='conv_1')

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), name='maxpool_1')

    # 2nd: convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify, name='conv_2')

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), name='maxpool_2')

    # 3rd convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=384, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify, name='conv_3')
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=384, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify, name='conv_4')
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify, name='conv_5')


    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=4096,
            name='hidden_6',
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected hidden layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.3),
            num_units=4096,
            name='hidden_7',
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0),
            num_units=2048,
            name='fc_A',
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0),
            num_units=96*96,
            name='fc_B',
            nonlinearity=lasagne.nonlinearities.sigmoid())

    # The output layer with 50% dropout on its inputs:
    # network = lasagne.layers.DenseLayer(
    #         lasagne.layers.dropout(network, p=.1),
    #         num_units=96*96,
    #         name='output_8',
    #         nonlinearity=lasagne.nonlinearities.sigmoid) # TODO: change

    return network

def iterate_minibatches(inputs, targets, batchsize):

    inputs_sz = len(inputs)
    #targets_sz = len(targets.get_value(borrow=True))

    for start_idx in range(0, inputs_sz - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def main(X_train, y_train, X_val, y_val, X_test, y_test, pixel, batch_size=150, num_epochs=100):
    n_batches_test = X_test.shape[0]/batch_size

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    # Create neural network model (depending on first command line parameter)
    network = build(input_var)

    # Construct a dict for all the layer for L2 reguarlization
    network_layers = lasagne.layers.get_all_layers(network)

    layers = {}
    for layer in network_layers:
        layer_name = layer.name
        if layer_name is not None:
            if type(layer) == lasagne.layers.Conv2DLayer:
                layers[layer] = 0.01
            elif type(layer) == lasagne.layers.DenseLayer:
                layers[layer] = 0.5

    l2_penalty = lasagne.regularization.regularize_layer_params_weighted(layers, lasagne.regularization.l2)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    train_prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(train_prediction, target_var)
    loss = loss.mean() + l2_penalty # add L2_penalty


    # Update the params with Nesterov Momentum to converge
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network)
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean()


    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], [train_prediction, loss], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    #val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    val_fn = theano.function([input_var, target_var], [test_prediction,test_loss])

    # Finally, launch the training loop.
    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size):
            inputs, targets = batch
            _,train_cost = train_fn(inputs, targets)
            print 'train error:', train_cost
            train_err += train_cost
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        #val_acc = 0
        val_batches = 0

        for batch in iterate_minibatches(X_val, y_val, batch_size):
            inputs, targets = batch
            _, err = val_fn(inputs, targets)
            val_err += np.mean(err)
            #val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        # print("  validation accuracy:\t\t{:.2f} %".format(
        #     val_acc / val_batches * 100))

    print 'done training...run test params'
    # conv_params = network_layers[6].get_params()
    # print conv_params[0].eval().shape
    # np.savez('conv_W.npz', conv_params[0].eval())
    # np.savez('conv_b.npz', conv_params[1].eval())

    # After training, we compute and print the test error:
    test_err = 0
    #test_acc = 0
    test_batches = 0

    test_preds = np.zeros((n_batches_test*batch_size, pixel[0]*pixel[1]))
    for batch in iterate_minibatches(X_test, y_test, batch_size):
        inputs, targets = batch
        y_hat, err = val_fn(inputs, targets)
        test_preds[test_batches*batch_size:(test_batches+1)*batch_size] = y_hat
        test_err += np.mean(err)
        test_batches += 1
    np.savez('test_preds_lenet5.npz', test_preds)

    # Optionally, you could now dump the network weights to a file like this:
    np.savez('model_lenet5.npz', lasagne.layers.get_all_param_values(network))

def prediction(X_test, y_test):
    # print 'predicting test set......'
    # # f_out = np.load(model)
    # # test_preds = f_out['arr_0']
    #
    # # load the parameters
    # p_out = np.load(model)
    # params = p_out['arr_0']
    #
    # trained = build_cnn(X_test)
    # lasagne.layers.set_all_param_values(trained, params)
    #
    # #Deterministic keyword disables stochastic behaviour such as dropout when set to True.
    # #This is useful because a deterministic output is desirable at evaluation time.
    # test_preds = lasagne.layers.get_output(trained,deterministic=True)
    # test_preds = test_preds.eval()
    # np.savez('test_pred_linear_cross_entropy_l2.npz', test_preds)

    # plot the output
    f_out= np.load('test_preds_lenet5.npz')
    test_preds = f_out['arr_0']
    for k in xrange(test_preds.shape[0]):
        print 'saving test output...', k
        y_hat = test_preds[k]
        # y_hat[y_hat < 0.0001] = 0
        # y_hat[y_hat >= 0.001] = 1
        plot_images(X_test[k], y_test[k], y_hat , k, alpha=0.1)


if __name__ == '__main__':
    theano.config.exception_verbosity='high'

    # load the data
    path='/home/quanle/Project_2015/Catheter_Localise/catheter_images_96.npz'
    #path='/home/quanle/Project_2015/Catheter_Localise/new_catheter_images_BIG.npz'
    #path='/Users/quale/Dropbox/Project/Code/Catheter_Localise/catheter_images_96.npz'
    model = 'model_params_linear_cross_entropy_l2.npz'

    pixel = (96,96)
    datasets = load_data(path, pixel)
    X_train, y_train = datasets[0]
    X_val, y_val = datasets[1]
    X_test, y_test = datasets[2]

    main(X_train, y_train, X_val, y_val, X_test, y_test, pixel, batch_size=150, num_epochs=100)
    #prediction(X_test, y_test)
