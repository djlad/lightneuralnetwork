# lightneuralnetwork
small neural network library written in python.

Uses backpropagation and gradient descent to adjust weights

## usage:
git clone url

``` python
import lightneuralnetwork
\#and gate inputs
example_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
\#and gate outputs
example_outputs = [
    [0],
    [0],
    [0],
    [1]
]

example_set = [example_inputs, example_outputs]
\#create neural network with 2 inputs, 12 hidden nodes, 1 output
net = lightneuralnetwork.Net([2, 12, 1], example_set)
\#run with inputs [1, 0] and [0, 0]
net.run([[1,0], [0,0]])
\#check examples.py for more details
```

## run tests:
cd lightneuralnetwork/src
./runtests.bat