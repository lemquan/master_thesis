__author__ = 'quale'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from util import plot_filters, load_data_for_model, plot_images, data_preprocess
import timeit
import cPickle
import sys
import matplotlib.pyplot as plt


''''
    This script trains a convolution network. The script contains the needed layers: convolutional, fully connected,
    and the SoftMax layer for prediction. It also includes a L2 regularlisation and dropout.

    There are functions called run_model and predict_model to train the CNN and predict. The run_model is modularized
    enough that the user can input additional layers of his discretion.
'''
rng = np.random.RandomState(1234) # good practice
sys.setrecursionlimit(10000)  # for pickle...
################## Import Activation Functions ######################
# http://cs231n.github.io/neural-networks-1/#actfun
def ReLu(x): #recommended use
    return T.maximum(x, 0.0)

####################### Define Network Layers ########################
class ConvPoolLayer():
    def __init__(self, rng, filter_shape, image_shape, poolsize=(2,2), activation_fn=ReLu):
        assert image_shape[1] == filter_shape[1]

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn

        # there are "num input feature maps * filter height * filter width"
        fan_in = np.prod(filter_shape[1:]) # number of inputs to each hidden unit

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))

        # initialize weights with random weights based on Xavier
        W_bound = np.sqrt(6.0/ (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                       dtype=theano.config.floatX), name='W_conv',
            borrow=True
        )

        #initialize the bias -- one bias per output feature map (filter)
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_conv', borrow=True)

        # store the parameters of this layer
        self.params = [self.W, self.b]

    def set_layer(self, input, input_dropout, mini_batch_size):
        self.input = input.reshape(self.image_shape)
        #self.input = input
        # convolve the inputs with filters
        conv_output = conv.conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        )

        # downsample each feature map/kernel using the max pool (CAN CHANGE TO AVG POOL)
        pooled_out = downsample.max_pool_2d(
            input = conv_output,
            ds = self.poolsize,
            ignore_border = True
        )

        # add the bias term to compute the activation function
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = self.activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in convolutional layer

class FullyConnectedLayer():
    def __init__(self, n_in, n_out, activation_fn=None, prob_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.prob_dropout = prob_dropout

        W_bound = np.sqrt(6.0/ (n_in + n_out))
        W_values = np.asarray(
            np.random.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)),
            dtype=theano.config.floatX
        )

        if activation_fn == theano.tensor.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value=W_values, name='W_FC', borrow=True)

        # initialize the bias
        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_FC', borrow=True)

        # store parameters of the FC layer
        self.params = [self.W, self.b]

    def set_layer(self, input, input_dropout, mini_batch_size):
        self.input = input.reshape((mini_batch_size, self.n_in))

        # calculate the output using a non linear activation function
        linear_output = (1-self.prob_dropout)*(T.dot(self.input, self.W) + self.b)
        self.output = self.activation_fn(linear_output)
        #self.y_out = T.argmax(self.output, axis=1) # logistic regression
        self.y_pred = T.nnet.sigmoid(self.output)

        # drop out
        new_input_dropout = input_dropout.reshape((mini_batch_size, self.n_in))
        self.input_dropout = _dropout_layer(new_input_dropout, self.prob_dropout)
        self.output_dropout = self.activation_fn(T.dot(self.input_dropout, self.W) + self.b)

    def accuracy(self,y):
        mse = T.mean((self.y_pred-y) ** 2)
        return mse

class RegressionLayer():
    def __init__(self, n_in, n_out, prob_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.prob_dropout = prob_dropout

        #initialize weights and biases
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W_sm',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name='b_sm',
            borrow=True
        )

        # store params of the model
        self.params = [self.W, self.b]

    def set_layer(self, input, input_dropout, mini_batch_size):
        # symoblic links to calculate teh prediction using  Regression
        # feedforward -- related to accuracy()
        self.input = input.reshape((mini_batch_size, self.n_in))
        self.output = (1-self.prob_dropout)*(T.dot(self.input, self.W) + self.b) # prediction output
        #self.y_pred = T.argmax(self.output, axis=1)
        self.y_pred = T.nnet.sigmoid(self.output)

        # dropout
        new_input_dropout = input_dropout.reshape((mini_batch_size, self.n_in))
        self.input_dropout = _dropout_layer(new_input_dropout, self.prob_dropout)
        self.output_dropout = T.dot(self.input_dropout, self.W) + self.b


    # compute the cost function
    def cost(self, net):
        err_matrix = (net.y -self.output_dropout) ** 2
        return T.sum(T.sum(err_matrix, axis=1))

    def accuracy(self, y):
        '''
        This is the feed forward part of the neural network. self.y_pred is the prediction
        from the test/train/valid data
        :param y: the true label
        :return:
        '''
        # calculate the loss function
        mse = T.mean((self.y_pred-y) ** 2)
        return self.output, mse


#################### Define the CNN Network ###########################
class CNN_Network():

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]

        # set up symoblic links
        self.x = T.tensor4('x')
        self.y = T.matrix('y')

        # set up the layers
        init_layer = self.layers[0] # the first layer (input image) - ConvPoolLayer
        init_layer.set_layer(self.x, self.x,self.mini_batch_size)
        for j in xrange(1, len(self.layers)): # start on the second layer
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_layer(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)

        # dropout
        self.output = self.layers[-1].output # last layer is output
        self.output_dropout = self.layers[-1].output_dropout


    def SGD(self, training_data, valid_data, test_data, n_epochs, mini_batch_size, lmbda=0.05, alpha=0.001):
        print
        train_x, train_y = training_data
        valid_x, valid_y = valid_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation, and test
        n_train_batches = train_x.get_value(borrow=True).shape[0] / mini_batch_size
        n_valid_batches = valid_x.get_value(borrow=True).shape[0] / mini_batch_size
        n_test_batches = test_x.get_value(borrow=True).shape[0] / mini_batch_size

        # define the L2 regularisation
        L2_norm = 0.5*sum([(layer.W **2).sum() for layer in self.layers])

        # define the cost function
        cost = self.layers[-1].cost(self) + (lmbda * L2_norm/n_train_batches)
        grads = T.grad(cost, self.params)

        # define the updates for params
        updates = [
            (param_i, param_i - alpha*grad_i) for param_i, grad_i in zip(self.params,grads)
        ]

        # # Nestrovo Momentum
        # momentum = 0.9
        # updates = []
        # for param in self.params:
        #     param_update = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))
        #     updates.append((param, param + param_update))
        #     eval_param = param + momentum * param_update
        #     updates.append((param_update, momentum*param_update-alpha* \
        #                     T.grad(cost, eval_param)))

        # define the functions for SGD
        idx = T.lscalar()
        train_model = theano.function(
            [idx],
            cost,
            updates=updates,
            givens={
                self.x: train_x[idx*self.mini_batch_size : (idx+1)*self.mini_batch_size],
                self.y: train_y[idx*self.mini_batch_size : (idx+1)*self.mini_batch_size],
            }
        )
        valid_model = theano.function(
            inputs=[idx],
            outputs=self.layers[-1].accuracy(self.y),
            givens={
                self.x: valid_x[idx*self.mini_batch_size : (idx+1)*self.mini_batch_size],
                self.y: valid_y[idx*self.mini_batch_size : (idx+1)*self.mini_batch_size],
            }
        )

        # training the model with SGD optimization
        patience = 100
        patience_increase = 2
        improvement_threshold = 0.0995
        validation_freq = min(n_train_batches, patience/2)
        best_validation = np.inf
        final_cost = []
        epoch = 0
        done_looping = False

        #plot the cost
        plt.figure()
        plt.title('Cost Function Per Epoch')
        #plt.axis([0, n_epochs, 0, 1.5])
        # plt.ion()
        # plt.show()
        f= '/home/quanle/Project_2015/Catheter_Localise/figs/' #TODO: change for the right folder
        #f = '/Users/quale/Dropbox/Project/Code/figs/'

        print 'CNN with regression'
        start_time = timeit.default_timer()
        while (epoch < n_epochs) and (not done_looping):
            print '---------------- epoch ', epoch, '------------------'
            cost_array = []
            for mini_batch_idx in xrange(n_train_batches):
                mini_batch_cost = train_model(mini_batch_idx)
                cost_array.append(mini_batch_cost)

                # increase iteration number
                iter = epoch* n_train_batches + mini_batch_idx
                print 'cost:', mini_batch_cost

                if (iter+1) % validation_freq == 0:
                    layer = theano

                    valid_err = []
                    for i in xrange(n_valid_batches):
                        _,mse = valid_model(i)
                        valid_err.append(mse)
                    valid_accur = np.mean(valid_err)
                    print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        mini_batch_idx + 1,
                        n_train_batches,
                        valid_accur * 100.
                    )
                    )
                    if valid_accur < best_validation: # need to fix this? should it be >=
                        if valid_accur < best_validation * improvement_threshold:
                            patience = max(patience, iter*patience_increase)

                        best_validation = valid_accur
                        best_iter = iter

                        # save the optimizd network and the params
                        with open('best_params_reg_cnn.pkl', 'wb') as save_file:
                            cPickle.dump(self, save_file, protocol=cPickle.HIGHEST_PROTOCOL)

                        # predict test set
                        y_hats = np.zeros((n_test_batches*mini_batch_size, 96*96))
                        for i in xrange(n_test_batches):
                            output, _ = valid_model(i)
                            y_hats[i*mini_batch_size: (i+1)*mini_batch_size] = output
                        print 'saving predictions from test'
                        with open('y_hats.pkl', 'wb') as save_file:
                            cPickle.dump(y_hats, save_file, protocol=cPickle.HIGHEST_PROTOCOL)

                if patience <= iter:
                    done_looping = True
                    break
            # avg the cost
            final_cost.append(np.mean(cost_array))
            x = epoch * np.ones(len(cost_array))
            plt.plot(x, cost_array, 'b')
            plt.draw()

            # # TODO: change for small or big ds
            # plot_filters(f,self, 0, 3,2, epoch)
            # plot_filters(f,self, 1, 8,2, epoch)

            # Small data set
            #plot_filters(f, self, 0, 4,5, epoch)
            #plot_filters(f, self, 1, 8,5, epoch)
            epoch += 1 # increment the counter
        # end of while loop

        plt.savefig(f + 'cost.png')
        plt.close()

        plt.figure()
        plt.title('Average Cost Function Per Epoch')
        plt.plot(range(len(final_cost)), final_cost)
        plt.savefig(f + 'avg_cost.png')
        end_time = timeit.default_timer()
        total_time = (end_time - start_time) / 60.
        print (('total time to run %.2fm') % total_time)

######################## Helper functions ###############################
def load_data(path):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    print '... loading data'
    train_input, train_label, valid_input, valid_label, test_input, test_label \
                                                    = load_data_for_model(path)


    train_input, mu = data_preprocess(train_input, mean=None)
    valid_input, _ = data_preprocess(valid_input, mu)
    test_input, _ = data_preprocess(test_input, mu)

    train_input = np.reshape(train_input, (-1, 1, 96, 96))
    valid_input = np.reshape(valid_input, (-1, 1, 96, 96))
    test_input = np.reshape(test_input, (-1, 1, 96, 96))

    print train_input.shape, train_label.shape
    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_input, test_label)
    valid_set_x,valid_set_y = shared_dataset(valid_input, valid_label)

    train_set_x, train_set_y = shared_dataset(train_input,train_label)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def _dropout_layer(layer, p):
    '''
    The dropout layer
    :param layer: the layer to drop
    :param p: the probability of a the nodes being dropped
    :return:
    '''
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

    # prob = 1- p
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    dropped_layer = layer * T.cast(mask, theano.config.floatX)
    return dropped_layer


###################### Train and Test the model ######################
def run_model(n_epochs, mini_batch_size, ds_path):
    datasets = load_data(ds_path)
    training_data = datasets[0]
    valid_data = datasets[1]
    test_data = datasets[2]

    alpha=0.01 # learning rate

    pixel = (224, 224)
    dropouts = [0.01, 0.01, 0.5] #TODO: change params of dropout
    # use with image shape 224,224 # LeCunn5
    # net = CNN_Network([
    #     ConvPoolLayer(rng, filter_shape=(6, 1, 5, 5),
    #                   image_shape=(mini_batch_size, 1, 224, 224),
    #                   poolsize=(2, 2),
    #                   activation_fn=ReLu),
    #     ConvPoolLayer(rng, filter_shape=(16, 6, 7,7),
    #                     image_shape=(mini_batch_size, 6, 110,110),
    #                     poolsize=(2,2),
    #                     activation_fn=ReLu),
    #     FullyConnectedLayer(n_in=16*52*52, n_out=500, prob_dropout=dropouts[0]),
    #     RegressionLayer(n_in=500, n_out=224*224, prob_dropout=dropouts[1])],
    #     mini_batch_size)
    net = CNN_Network([
        ConvPoolLayer(rng, filter_shape=(32, 1, 3, 3),
                      image_shape=(mini_batch_size, 1, 96, 96),
                      poolsize=(2, 2),
                      activation_fn=ReLu),
        ConvPoolLayer(rng, filter_shape=(64, 32, 2,2),
                        image_shape=(mini_batch_size, 32, 47,47),
                        poolsize=(2,2),
                        activation_fn=ReLu),
        ConvPoolLayer(rng, filter_shape=(128, 64, 2,2),
                        image_shape=(mini_batch_size, 64, 23,23),
                        poolsize=(2,2),
                        activation_fn=ReLu),
        ConvPoolLayer(rng, filter_shape=(256, 128, 2,2),
                        image_shape=(mini_batch_size, 128, 11,11),
                        poolsize=(2,2),
                        activation_fn=ReLu),
        ConvPoolLayer(rng, filter_shape=(512, 256, 2,2),
                        image_shape=(mini_batch_size, 256, 5,5),
                        poolsize=(2,2),
                        activation_fn=ReLu),
        FullyConnectedLayer(n_in=512*2*2, n_out=600, activation_fn=ReLu,prob_dropout=dropouts[0]),
        FullyConnectedLayer(n_in=600, n_out=600, activation_fn=ReLu, prob_dropout=dropouts[0]),
        RegressionLayer(n_in=600, n_out=96*96, prob_dropout=dropouts[0])],
        mini_batch_size)

    net.SGD(training_data, valid_data, test_data, n_epochs, mini_batch_size, alpha)
    return net

def predict_model(mini_batch_size, path, filename='best_params_reg_cnn.pkl'):
    # feed forward
    dataset = load_data(path)
    test_x, test_y = dataset[2]

    # get the values of the test set to calculate MSE
    test_x_vals = test_x.get_value(borrow=True)
    test_y_vals = test_y.get_value(borrow=True)

    n_test_batches = test_x_vals.shape[0]/mini_batch_size
    #
    # network = cPickle.load(open(filename)) # load the network
    # idx = T.lscalar()
    # test_model = theano.function(
    #     inputs=[idx],
    #     outputs=network.layers[-1].accuracy(network.y),
    #     givens={
    #         network.x: test_x[idx*mini_batch_size : (idx+1)*mini_batch_size],
    #         network.y: test_y[idx*mini_batch_size : (idx+1)*mini_batch_size],
    #     }
    # )
    #
    # all_mse = np.zeros(n_test_batches)
    # y_hats = np.zeros((n_test_batches*mini_batch_size, test_x_vals.shape[2]* test_x_vals.shape[3]))
    # for i in xrange(n_test_batches):
    #     output, err = test_model(i)
    #     print output.shape
    #     all_mse[i] = err
    #     y_hats[i*mini_batch_size: (i+1)*mini_batch_size] = output
    #
    # with open('y_hats.pkl', 'wb') as save_file:
    #     cPickle.dump(y_hats, save_file, protocol=cPickle.HIGHEST_PROTOCOL)

    preds = cPickle.load(open('/Users/quale/Dropbox/Project/Code/Catheter_Localise/output/y_hats.pkl'))
    preds
    # plot results
    # for k in xrange(len(y_hats)):
    #     print 'saving output...', k
    #     plot_images(test_x_vals[k], test_y_vals[k],y_hats[k], k, pixels=(96,96), alpha=0.5)

    #print 'Accuracy:', np.mean(all_mse)*100, '%'

if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    theano.config.optimizer='fast_compile'
    theano.config.floatX = 'float32'
    # parameters
    epochs = 250
    mini_batch_size = 100

    # load the data # TODO: change for the right folder

    path='/home/quanle/Project_2015/Catheter_Localise/catheter_images_96.npz'
    #path = '/Users/quale/Dropbox/Project/Code/Catheter_Localise/catheter_images_96.npz'
    run_model(epochs, mini_batch_size, path)
    #predict_model(mini_batch_size, path) # TODO: comment out when running training for memory conservation









