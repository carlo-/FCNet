# FCNet
Python implementation of a fully connected neural network.

Training is performed exclusively on CPU, and is implemented as Mini-batch Gradient Descent.\
All settings of the descent can be customized for tuning.

## Technical details
- **Initialization:** He
- **Activation function:** ReLU
- **Cost function:** Cross-entropy loss + regularization term
- **Regularization:** L2 with customizable factor (lambda)
- **Output:** Softmax of the scores at the final layer

## Network configuration
The `Net` object must be initialized with two arguments, an array containing the number of nodes at each layer (including input and output sizes), and a dictionary representing the configuration of the gradient descent.
```Python
net_sizes = [3072, 50, 10] # [input, ...hidden, output]
descent_config = {} # empty for simplicity
net = Net(net_sizes, descent_config)
```
In the example above, the initialized network has an input of `3072` dimensions, one single hidden layer with `50` nodes, and an output of `10` dimensions.

## Descent configuration
Below all settings for the gradient descent with their default values:
```Python
descent_config = {
    'eta': 0.01, # learning rate
    'batch_size': 100, # size of each batch to be used for training
    'epochs': 40, # number of epochs
    'gamma': 0.0, # momentum factor
    'decay_rate': 1.0, # rate of decay of eta
    'lambda': 0.0, # regularization factor
    'batch_normalize': True, # whether or not batch normalization should be used
    'plateau_guard': None, # if the speed of descent becomes greater than this value, eta is divided by 10.0
    'overfitting_guard': None, # if the speed of descent becomes greater than this value, training is aborted
    'output_folder': None # if specified, the model will be exported here after each epoch 
}
```

## Full example
```Python
np.random.seed(42)

X, Y, y = dataset.load_multibatch_cifar10()
X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch')

K, d = (Y.shape[0], X.shape[0])
net_sizes = [d, 50, 30, K]

descent_config = {
    'eta': 0.015,
    'batch_size': 100,
    'epochs': 20,
    'gamma': 0.6,
    'decay_rate': 0.93,
    'lambda': 0.0001,
    'batch_normalize': True,
    'plateau_guard': -0.001,
    'overfitting_guard': 0.0,
    'output_folder': '../model/'
}

net = Net(net_sizes, descent_config)
net.train(X, Y, X_test, Y_test)

training_accuracy = net.compute_accuracy(X, y)
test_accuracy = net.compute_accuracy(X_test, y_test)
```


## License
This project is released under the MIT license. See `LICENSE` for more information.
