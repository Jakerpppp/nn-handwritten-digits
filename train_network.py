from os.path import join
from neural_network import BasicNetwork
from improved_neural_network import ImprovedNetwork, CrossEntropyCost, QuadraticCost
import mnist_loader
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Function to preprocess the data
def preprocess_data(data):
    # Set any pixel value > 0 to 1 using numpy array manipulation
    processed_data = []
    for x, y in data:
        x = np.where(x > 0, 1, 0)
        processed_data.append((x, y))
    return processed_data

# Preprocess the training data
training_data = preprocess_data(training_data)



def train_first_neural_network():
    net = BasicNetwork([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data)


def train_improved_neural_network():
    net = ImprovedNetwork([784, 30, 10], cost=CrossEntropyCost)
    net.SGD(
    training_data=training_data,
    epochs=100,
    mini_batch_size=10,
    eta=3.0,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_evaluation_cost=True,
    monitor_training_accuracy=True,
    monitor_training_cost=True)
    net.save(join("networks", "all_white_inputs.json"))

train_improved_neural_network()
