# FCNet
Python implementation of a fully connected neural network.

Training is performed exclusively on CPU, and is implemented as Mini-batch Gradient Descent.\
All settings of the descent can be customized for tuning.

## Technical details:
- **Initialization:** He
- **Activation function:** ReLU
- **Cost function:** Cross-entropy loss + regularization term
- **Regularization:** L2 with customizable factor
- **Output:** Softmax of the scores at the final layer

## Network configuration:
The `Net` object must be initialized with two arguments, an array containing the number of nodes at each layer (including input and output sizes), and a dictionary representing the configuration of the gradient descent.
```Python
net_sizes = [3072, 50, 10] # [input, ...hidden, output]
descent_config = {} # empty for simplicity
net = Net(net_sizes, descent_config)
```
In the example above, the initialized network has an input of `3072` dimensions, one single hidden layer with `50` nodes, and an output of `10` dimensions.

## Descent configuration:

## License
This project is released under the MIT license. See `LICENSE` for more information.
