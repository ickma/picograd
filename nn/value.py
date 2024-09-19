import math


class BaseValue:
    def __init__(self, value):
        self.value = value
        self.out = None
        self.grad = 0
        self.backwards = []
        self.backward_times = 0

    def __add__(self, other):  # self + other
        """

        Addition operation for Parameters:

        Let f(x, y) = x + y, where x and y are Parameter objects.

        Partial derivatives:
        ∂f/∂x = 1
        ∂f/∂y = 1

        Proof of ∂f/∂x using limit definition:
        lim[Δx→0] (f(x + Δx, y) - f(x, y)) / Δx
        = lim[Δx→0] ((x + Δx + y) - (x + y)) / Δx
        = lim[Δx→0] Δx / Δx
        = 1

        Similarly, ∂f/∂y = 1
        """
        other = other if isinstance(other, BaseValue) else BaseValue(other)
        out = BaseValue(self.value + other.value)
        self.out = out

        out.backwards.extend([(self, 1), (other, 1)])
        return out

    def __mul__(self, other):  # self * other
        """
        Multiplication operation for Parameters:

        Let f(x, y) = x * y, where x and y are Parameter objects.

        Partial derivatives:
        ∂f/∂x = y
        ∂f/∂y = x

        Proof of ∂f/∂x using limit definition:
        lim[Δx→0] (f(x + Δx, y) - f(x, y)) / Δx
        = lim[Δx→0] ((x + Δx) * y - x * y) / Δx
        = lim[Δx→0] (xy + Δxy - xy) / Δx
        = lim[Δx→0] (Δxy) / Δx
        = lim[Δx→0] y
        = y
        """
        other = other if isinstance(other, BaseValue) else BaseValue(other)
        out = BaseValue(self.value * other.value)
        out.backwards.extend([(self, other.value), (other, self.value)])
        return out

    def __pow__(self, other):  # self ** other
        """
        Power operation for Parameters:

        Let f(x, y) = x^y, where x and y are Parameter objects.

        Partial derivatives:
        ∂f/∂x = y * x^(y-1)
        ∂f/∂y = x^y * ln(x)

        Proof of ∂f/∂x using limit definition:
        lim[Δx→0] (f(x + Δx, y) - f(x, y)) / Δx
        = lim[Δx→0] ((x + Δx)^y - x^y) / Δx
        = lim[Δx→0] (x^y * (1 + Δx/x)^y - x^y) / Δx
        = lim[Δx→0] (x^y * (1 + y * Δx/x + O(Δx^2))) - x^y) / Δx
        = lim[Δx→0] (x^y * (y * Δx/x + O(Δx^2))) / Δx
        = lim[Δx→0] (x^y * y * Δx/x) / Δx
        = lim[Δx→0] (x^y * y)
        = y * x^(y-1)
        """
        other = other if isinstance(other, BaseValue) else BaseValue(other)
        out = BaseValue(self.value ** other.value)
        self.out = out
        out.backwards.extend(
            [
                (self, other.value * self.value ** (other.value - 1))
            ]
        )
        return out

    def relu(self):
        out = BaseValue(0 if self.value < 0 else self.value)
        self.out = out
        out.backwards.extend([(self, 1 if self.value > 0 else 0)])
        return out

    def sigmoid(self):
        out = BaseValue(1 / (1 + math.exp(-self.value)))
        self.out = out
        out.backwards.extend([(self, out.value * (1 - out.value))])
        return out

    def log(self):
        out = BaseValue(math.log(self.value))
        self.out = out
        out.backwards.extend([(self, 1 / self.value)])
        return out

    def exp(self):
        out = BaseValue(math.exp(self.value))
        self.out = out
        out.backwards.extend([(self, out.value)])
        return out

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * (other ** -1)

    def __radd__(self, other):  # other + self
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rtruediv__(self, other):  # other / self
        return other * (self ** -1)

    def __repr__(self):
        return f"Parameter({self.value})"

    def __abs__(self):
        out = BaseValue(abs(self.value))
        out.backwards.extend([(self, 1 if self.value >= 0 else -1)])
        return out

    def backward(self):
        """
        Performs backward propagation (backpropagation) through the computational graph.
        The method does the following:
        1. Sets the gradient of the current node to the input gradient.
        2. Iterates through all connected nodes (parents) in the computational graph.
        3. For each parent node, it:
           a. Multiplies the current gradient by the partial derivative w.r.t. that node.
           b. Accumulates this product into the parent node's gradient.
           c. Recursively calls backward() on the parent node.

        This process effectively implements the chain rule of calculus, allowing
        gradients to flow backward through the entire computational graph.
        """

        for node, partial_derivative in self.backwards:
            node.grad += self.grad * partial_derivative
            node.backward()
            if not isinstance(node, Parameter):
                node.zero_grad()

    def zero_grad(self):
        """
        Resets the gradient of this parameter to zero.

        This method is typically called before starting a new round of backpropagation
        to clear out any previously accumulated gradients. It ensures that gradients
        from previous computations don't interfere with the current computation.
        """
        self.grad = 0


class Parameter(BaseValue):
    def __init__(self, value):
        super().__init__(value)

    def __repr__(self):
        return f"Parameter({self.value})"
