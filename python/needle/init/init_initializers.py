import math
from .init_basic import *

# =====
# xavier initialization
# suitable for linear function like tanh, sigmoid
#   which are approximate linear around the 'zero' and symmetric
# =====

def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    shape = (fan_in, fan_out)
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(*shape, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    shape = (fan_in, fan_out)
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION


# =====
# kaiming initialization
# suitable for linear function like ReLU
# it only focus the input dimension size, which decides the variance
# =====


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = math.sqrt(6 / fan_in)
    if shape is None:
        shape = (fan_in, fan_out)
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2 / fan_in)
    shape = (fan_in, fan_out)
    return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION