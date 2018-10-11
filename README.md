# SingleLayerPerceptronNetwork
creates a simple single layer perceptron neural network to predict an output from a set of inputs
<img src="neuralnetgif1.gif" height="300" />
<img src="neuralnetgif2.gif" height="300" />

I used this artificial neural network to solve a simple task. Given the inputs and outputs below:

| Inputs        | Output  |
| ------------- |:-------:|
| 0 0 1 1       | 0       |
| 1 1 1 1       | 1       |
| 1 1 0 1       | 1       |
| 0 1 1 1       | 0       |

I wanted the neural network to be able to predict what the output should be for the new input:

1 0 0 1

The output should result in 1 if the inputs first value is 1. Otherwise, the output is 0

Training the network 1000 times using backpropagation, the result of the new input was:
0.9871252 (almost 1.0)
