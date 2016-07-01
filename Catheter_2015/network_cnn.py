__author__ = 'quale'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from util import plot_filters, load_data_for_model
import timeit
import cPickle
import matplotlib.pyplot as plt

################## Import Activation Functions ######################
# http://cs231n.github.io/neural-networks-1/#actfun
from theano.tensor.nnet import sigmoid # avoid using

from theano.tensor import tanh # worst than ReLu

def ReLu(x): #recommended use
    return T.maximum(x, 0.0)

def LeakyReLu(x, alpha=3.0): # best case
    return T.maximum(x, x*(1.0 / alpha))


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


    def set_layer(self, input, mini_batch_size):
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

class FullyConnectedLayer():
    def __init__(self, n_in, n_out, activation_fn=ReLu):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn

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

    def set_layer(self, input, mini_batch_size):
        self.input = input.reshape((mini_batch_size, self.n_in))

        # calculate the output using a non linear activation function
        linear_output = T.dot(self.input, self.W) + self.b
        self.output = (
            linear_output if self.activation_fn is None
            else self.activation_fn(linear_output)
        )
        # find the max value for prediction
        self.y_out = T.argmax(self.output, axis=1)

    def accuracy(self,y):
        return T.mean(T.eq(y, self.y_out))

class SoftMaxLayer():
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

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

    def set_layer(self, input, mini_batch_size):
        # symoblic links to calculate teh prediction using SoftMax Regression
        # feedforward -- related to accuracy()
        self.input = input.reshape((mini_batch_size, self.n_in))
        self.output = T.nnet.softmax(T.dot(self.input, self.W) + self.b) # prediction output
        self.y_pred = T.argmax(self.output, axis=1)

    # compute the cost function
    def negative_log_likelihood(self, net): #pass the Network object in
        res = -T.mean(T.log(self.output)[T.arange(net.y.shape[0]), net.y])
        return res

    def accuracy(self, y):
        '''
        This is the feed forward part of the neural network. self.y_pred is the prediction
        from the test/train/valid data
        :param y: the true label
        :return:
        '''
        err = T.mean(T.neq(self.y_pred, y))
        return (self.y_pred, err)



#################### Define the CNN Network ###########################
class CNN_Network():

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]

        # set up symoblic links
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        # set up the layers
        init_layer = self.layers[0] # the first layer (input image) - ConvPoolLayer
        init_layer.set_layer(self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)): # start on the second layer
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_layer(prev_layer.output, self.mini_batch_size)


        self.output = self.layers[-1].output # last layer is output

        # L2 regularization
        self.L2 = 0.0
        for j in xrange(1,len(self.layers)):
            self.L2 = self.L2 + (self.layers[j].W ** 2).sum()


    def SGD(self, training_data, valid_data, test_data, n_epochs, mini_batch_size, lmbda=0.0001, alpha=0.001):
        train_x, train_y = training_data
        valid_x, valid_y = valid_data

        # compute number of minibatches for training, validation, and test
        n_train_batches = train_x.get_value(borrow=True).shape[0] / mini_batch_size
        n_valid_batches = valid_x.get_value(borrow=True).shape[0] / mini_batch_size

        # define the cost function
        cost = self.layers[-1].negative_log_likelihood(self) + (lmbda * self.L2)
        grads = T.grad(cost, self.params)
        updates = [
            (param_i, param_i - alpha*grad_i) for param_i, grad_i in zip(self.params,grads)
        ]

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
        improvement_threshold = 0.995
        validation_freq = min(n_train_batches, patience/2)
        best_validation = np.inf
        final_cost = []
        epoch = 0
        done_looping = False


        # plot the cost
        plt.figure()
        plt.title('Cost Function Per Epoch')
        plt.axis([0, n_epochs, 0, 2.5])
        plt.ion()
        plt.show()

        print 'training model....'
        start_time = timeit.default_timer()
        while (epoch < n_epochs) and (not done_looping):
            print '---------------epoch ', epoch, '----------------'
            cost_array = []
            for mini_batch_idx in xrange(n_train_batches):
                mini_batch_cost = train_model(mini_batch_idx)
                cost_array.append(mini_batch_cost)

                # increase iteration number
                iter = epoch* n_train_batches + mini_batch_idx
                print 'cost:', mini_batch_cost

                # if iter % 100 == 0:
                #     print 'training @ iter =', iter

                if (iter+1) % validation_freq == 0:
                    valid_err = []
                    for i in xrange(n_valid_batches):
                        _,e = valid_model(i)
                        valid_err.append(e)
                    valid_accur = np.mean(valid_err)

                    if valid_accur < best_validation: # need to fix this? should it be >=
                        if valid_accur < best_validation * improvement_threshold:
                            patience = max(patience, iter*patience_increase)

                        best_validation = valid_accur
                        best_iter = iter

                        # save the optimizd network and the params
                        with open('best_params_cnn.pkl', 'wb') as save_file:
                            cPickle.dump(self, save_file, protocol=cPickle.HIGHEST_PROTOCOL)

                if patience <= iter:
                    done_looping = True
                    break
            # avg the cost
            final_cost.append(np.mean(cost_array))
            x = epoch * np.ones(len(cost_array))
            plt.plot(x, cost_array, 'b')
            plt.draw()

            #plot_filters(self, 0, 3,2, epoch)
            #plot_filters(self, 1, 8,2, epoch)
            #plot_filters(self, 0, 4,5, epoch)
            #plot_filters(self, 1, 8,5, epoch)
            epoch += 1 # increment the counter
        # end of while loop

        f = '/home/quanle/Project_2015/figs/'
        f = '/Users/quale/Dropbox/Project/Code/figs'
        plt.savefig(f + 'cost.png')
        plt.close()

        plt.figure()
        plt.title('Average Cost Function Per Epoch')
        plt.plot(range(len(final_cost)), final_cost)
        plt.savefig(f + 'avg_cost.png')
        end_time = timeit.default_timer()
        total_time = (end_time - start_time) / 60.
        print (('total time to run %.2fm') % total_time)

def load_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    print '... loading data'
    # load the data
    path='/home/quanle/Project_2015/catheter_images_BIG.npz'
    #path = 'catheter_images_SM.npz'
    #path = '/Users/quale/Dropbox/Project/Code/Numpy_Load/catheter_images_BIG.npz'
    train_input, train_label, valid_input, valid_label, test_input, test_label \
                                                    = load_data_for_model(path)

    #print train_input.shape, valid_input.shape, test_input.shape
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

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_input, test_label)
    valid_set_x,valid_set_y = shared_dataset(valid_input, valid_label)

    train_set_x, train_set_y = shared_dataset(train_input,train_label)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def run_model(n_epochs, mini_batch_size):
    rng = np.random.RandomState(1234) # good practice

    datasets = load_data()
    training_data = datasets[0]
    valid_data = datasets[1]
    test_data = datasets[2]

    alpha=0.001

    pixel = (224, 224)
    # LetNet structure - use with image 28,28
    '''net = CNN_Network([
            ConvPoolLayer(rng, filter_shape=(20, 1, 5, 5),
                          image_shape=(mini_batch_size, 1, 28, 28),
                          poolsize=(2, 2),
                          activation_fn=ReLu),
            ConvPoolLayer(rng, filter_shape=(40, 20, 5,5),
                            image_shape=(mini_batch_size, 20, 12,12),
                            poolsize=(2,2),
                          activation_fn=ReLu),
            FullyConnectedLayer(n_in=40*4*4, n_out=100),
            SoftMaxLayer(n_in=100, n_out=2)],
            mini_batch_size) '''

    # use with image shape 224,224
    net = CNN_Network([
        ConvPoolLayer(rng, filter_shape=(6, 1, 5, 5),
                      image_shape=(mini_batch_size, 1, 224, 224),
                      poolsize=(2, 2),
                      activation_fn=tanh),
        ConvPoolLayer(rng, filter_shape=(16, 6, 7,7),
                        image_shape=(mini_batch_size, 6, 110,110),
                        poolsize=(2,2),
                        activation_fn=tanh),
        FullyConnectedLayer(n_in=16*52*52, n_out=100),
        SoftMaxLayer(n_in=100, n_out=2)],
        mini_batch_size)

    net.SGD(training_data, valid_data, test_data, n_epochs, mini_batch_size, alpha)
    return net

def predict_model(mini_batch_size, filename='best_params_cnn.pkl'):
    network = cPickle.load(open(filename))
    idx = T.lscalar()

    # feed forward
    dataset = load_data()
    test_x, test_y = dataset[2]
    n_test_batches = test_x.get_value(borrow=True).shape[0]/mini_batch_size

    test_model = theano.function(
        inputs=[idx],
        outputs=network.layers[-1].accuracy(network.y),
        givens={
            network.x: test_x[idx*mini_batch_size : (idx+1)*mini_batch_size],
            network.y: test_y[idx*mini_batch_size : (idx+1)*mini_batch_size],
        }
    )

    y_hat = []
    err = []
    for i in xrange(n_test_batches):
        y,e = test_model(i)
        y_hat.extend(y)
        err.append(e)

    print 'test error is:', 100.*np.mean(err)

if __name__ == '__main__':
    theano.config.exception_verbosity='high'

    # parameters
    epochs = 300
    mini_batch_size = 60
    run_model(epochs, mini_batch_size)
    #predict_model(mini_batch_size)









