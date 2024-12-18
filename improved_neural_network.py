import numpy as np
import random
import time
import json
import sys

class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a))) #Return the cost associated with an output "a" and desired output "y"
    
    @staticmethod
    def delta(z, a, y):
        return (a-y) #Return the error delta from the output layer.
    
class QuadraticCost: #Old Cost Function
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2 #Return the cost associated with an output "a" and desired output "y"
    
    @staticmethod
    def delta(z, a, y):
        return (a-y)*sigmoid_prime(z) #Return the error delta from the output layer.

class ImprovedNetwork:
    def __init__(self, sizes, cost=CrossEntropyCost) -> None:
        self.num_layers = len(sizes) 
        self.sizes = sizes #Size of each layer
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])] #Mean 0 s.d 1/Sqrt(n) 
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]] #Mean 0, S.d 1

    def large_weight_initializer(self): #Old Weights for testing purposes
        self.weights= [np.random.randn(y, x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example, if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            time2 = time.time()
            print("Time taken: %s seconds" % (time2-time1))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
            print()
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] #Regularized Weights
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it flags whether we need to convert between the different
        representations.  It may seem strange to use different representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets. These are different types of computations, and using different
        representations speeds things up.  More details on the representations can be found in mnist_loader.load_data_wrapper.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x,y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x,y) in data]
        return sum(int(x==y) for (x,y) in results)
    
    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x,y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) #Regularization term
        return cost
    
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()



def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = ImprovedNetwork(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

    
