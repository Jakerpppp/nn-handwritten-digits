from os.path import join
from load_data import MnistDataloader
import numpy as np
import random

input_path = 'dataset'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data() 


class Network:
    def __init__(self, sizes) -> None:
        self.num_layers = len(sizes) 
        self.sizes = sizes #Size of each layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Initialize biases randomly omit input layer
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #Initialize weights randomly omit input layer

    def feedforward(self, a): #Return the output of the network if "a" is input. Applys to each Layer (2 and 3)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for tracking progress, but slows things down substantially."""
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data) #Shuffle the training data
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] #Divide the training data into mini_batches
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) #Update the weights and biases for each mini_batch (One Step of SGD)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta" is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] #Original W - (Learning Rate/Mini_batch_size)*Gradient for each weight
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)] #Original B - (Learning Rate/Mini_batch_size)*Gradient for each bias


    
def sigmoid(z):
    1 / (1 + np.exp(-z))
        


net = Network([784, 16, 10])
print(net.weights[1])


