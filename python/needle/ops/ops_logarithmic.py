from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[Tuple[int, ...] | int] = None):
        if isinstance(axes, int):
            self.axes = (axes, )
        else:
            self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        '''
        log(
            sum(
                exp(
                    Z - max(Z, axes, keep_dim=True)
                )
            )
        ) + max(Z, axes, keep_dim=False)
        '''
        max_Z_origin = maximum(Tensor(Z, device=Z.device), axes=self.axes, keepdims=True).data.cached_data
        max_Z_reduce = maximum(Tensor(Z, device=Z.device), axes=self.axes).data.cached_data
        sum = summation(Tensor((Z - max_Z_origin.broadcast_to(Z.shape)).exp(), device=Z.device), axes=self.axes).data.cached_data
        return sum.log() + max_Z_reduce
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[Tuple[int, ...] | int]=None) -> Tensor:
    return LogSumExp(axes=axes)(a)

