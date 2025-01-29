"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # self.weight = (in_features, out_features)
        self.weight = Parameter(init.kaiming_uniform(fan_in=self.in_features, fan_out=self.out_features,
                                             device=device, dtype=dtype, requires_grad=True))
        self.has_bias = bias
        if self.has_bias is True:
            # self.bias = (1, out_features)
            self.bias = Parameter(init.kaiming_uniform(fan_in=self.out_features, fan_out=1,
                                             device=device, dtype=dtype, requires_grad=True).reshape((1, self.out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        z = X @ self.weight # z=(N, out_features)
        if self.has_bias:
            z = z + self.bias.broadcast_to(z.shape)
        return z
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = x
        for m in self.modules:
            output = m(output)
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n, k = logits.shape
        one_hot = init.one_hot(k, y, device=logits.device, dtype=logits.dtype) # (n, k)
        return (ops.logsumexp(logits, axes=(1,)).sum() - (one_hot * logits).sum()) / n
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x = (n_batch, n_features)
        if self.training:
            n_batch = x.shape[0]
            
            E_x = (x.sum(axes=(0,)) / n_batch) # E_x = (n_features)
            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * E_x).detach() # 防止积累过大的计算图和tensor
            E_x = E_x.reshape((1, -1)).broadcast_to(x.shape)

            Var_x = ((x - E_x) ** 2).sum(axes=(0,)) / n_batch
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * Var_x).detach()
            Var_x = Var_x.reshape((1, -1)).broadcast_to(x.shape)

            norm = (x - E_x) / ((Var_x + self.eps) ** 0.5)
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        # eval mode
        mean = self.running_mean.reshape((1, -1)).broadcast_to(x.shape)
        var = self.running_var.reshape((1, -1)).broadcast_to(x.shape)
        norm = (x - mean) / ((var + self.eps) ** 0.5)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x = (n_batch, n_features)
        n_features = x.shape[1]
        E_x = x.sum(axes=(1,)) / n_features # E_x = (n_batch)
        E_x = E_x.reshape((-1, 1)).broadcast_to(x.shape)
        Var_x = ((x - E_x) ** 2).sum(axes=(1,)) / n_features
        Var_x = Var_x.reshape((-1, 1)).broadcast_to(x.shape)
        norm = (x - E_x) / ((Var_x + self.eps) ** 0.5)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            ######################## How to generate the mask ########################
            #               `np.random.binomail()` or `init.randb()` ?
            # I think it's just because the implementation of the standard answer 
            #                       uses `randb()` that only it can pass the test
            # however, in fact it's definitely possible to use `binomial()` here,
            #          because it obeys the Bernoulli distribution, 
            #          but randb() produces the uniform distribution.
            ##########################################################################
            mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype)
            return x * mask / (1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
