from os.path import join
from load_data import MnistDataloader
import numpy as np

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

net = Network([784, 16, 10])
print(net.weights[1])


