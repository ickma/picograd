import random
from .value import Parameter


class Layer:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = [Parameter(random.uniform(-1, 1))
                        for _ in range(in_dim * out_dim)]
        self.biases = [Parameter(random.uniform(-1, 1))
                       for _ in range(out_dim)]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs):
        """
        Performs the forward pass of the layer.

        This method computes the output of the layer given an input. It applies
        the linear transformation y = Wx + b, where W is the weight matrix,
        x is the input vector, and b is the bias vector.

        Args:
            inputs (list): A list of input values, should have length equal to self.in_dim.

        Returns:
            list: The output values after applying the layer's transformation.
                  The length of this list will be equal to self.out_dim.

        Note:
            This implementation uses explicit loops for clarity. In practice,
            matrix operations would be more efficient for larger dimensions.
        """
        outputs = []
        for i in range(self.out_dim):
            output = 0
            for j in range(self.in_dim):
                output += self.weights[i * self.in_dim + j] * inputs[j]
            output = self.biases[i]+output
            outputs.append(output)
        return outputs
