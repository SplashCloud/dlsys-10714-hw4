"""Operator implementations."""

from numbers import Number
from typing import Any, Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

from ..needle_logger import global_logger

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

from .gradient_validator import get_validator

def validate_grad(op: TensorOp, out_grad: Tensor, node: Tensor, actual_grad: Any):
    validator = get_validator()
    validator.validate(op, out_grad, node, actual_grad)

def broadcast_inverse_op(before_bct_shape, after_bct_shape, grad: Tensor):
    '''
    use for compute the grad of the tensor which has been broadcast_to
    before_bct: the shape of origin tensor
    after_bct: the shape of result tensor
    grad: the grad which has same shape with after_bct
    '''
    expand = len(after_bct_shape) - len(before_bct_shape)
    axes_to_sum = tuple(i for i in range(expand))
    for i in range(len(after_bct_shape)-1, expand-1, -1):
        if before_bct_shape[i-expand] != after_bct_shape[i]:
            axes_to_sum += (i,)
    if axes_to_sum:
        return reshape(summation(grad, axes=axes_to_sum), before_bct_shape)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs_grad, rhs_grad = out_grad, out_grad
        lhs, rhs = node.inputs
        if lhs.shape != node.shape:
            lhs_grad = broadcast_inverse_op(lhs.shape, node.shape, lhs_grad)
        if rhs.shape != node.shape:
            rhs_grad = broadcast_inverse_op(rhs.shape, node.shape, rhs_grad)
        validate_grad(self, out_grad, node, (lhs_grad, rhs_grad))
        return lhs_grad, rhs_grad


def add(a: Tensor, b: Tensor) -> Tensor:
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        validate_grad(self, out_grad, node, out_grad)
        return out_grad


def add_scalar(a: Tensor, scalar) -> Tensor:
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        # Handle broadcasting: sum over dimensions that were broadcast
        lhs_grad = out_grad * rhs
        rhs_grad = out_grad * lhs
        
        # Sum over broadcast dimensions for lhs (similar to BroadcastTo gradient)
        if lhs.shape != node.shape:
            lhs_grad = broadcast_inverse_op(lhs.shape, node.shape, lhs_grad)
        
        # Sum over broadcast dimensions for rhs (similar to BroadcastTo gradient)
        if rhs.shape != node.shape:
            rhs_grad = broadcast_inverse_op(rhs.shape, node.shape, rhs_grad)
        
        validate_grad(self, out_grad, node, (lhs_grad, rhs_grad))
        return lhs_grad, rhs_grad


def multiply(a: Tensor, b: Tensor) -> Tensor:
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        validate_grad(self, out_grad, node, out_grad * self.scalar)
        return (out_grad * self.scalar,)


def mul_scalar(a: Tensor, scalar) -> Tensor:
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a: Tensor, b: Tensor) -> Tensor:
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        result = out_grad * self.scalar * (a ** (self.scalar - 1))
        validate_grad(self, out_grad, node, result)
        return result
        ### END YOUR SOLUTION


def power_scalar(a: Tensor, scalar) -> Tensor:
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        # Compute gradients: d(a/b)/da = 1/b, d(a/b)/db = -a/(b^2)
        lhs_grad = out_grad / b
        rhs_grad = out_grad * (-1) * a / (b ** 2)
        
        # Handle broadcasting: sum over dimensions that were broadcast
        # Sum over broadcast dimensions for lhs (similar to BroadcastTo gradient)
        if a.shape != node.shape:
            lhs_grad = broadcast_inverse_op(a.shape, node.shape, lhs_grad)
        
        # Sum over broadcast dimensions for rhs (similar to BroadcastTo gradient)
        if b.shape != node.shape:
            rhs_grad = broadcast_inverse_op(b.shape, node.shape, rhs_grad)
        
        validate_grad(self, out_grad, node, (lhs_grad, rhs_grad))
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def divide(a: Tensor, b: Tensor) -> Tensor:
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        result = out_grad / self.scalar
        validate_grad(self, out_grad, node, result)
        return result
        ### END YOUR SOLUTION


def divide_scalar(a: Tensor, scalar) -> Tensor:
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    '''
      axes: None or a 2-ele tuple
      None: exchange the last 2 dimensions
      tuple: exchange the specified 2 dimensions 
    '''
    def __init__(self, axes: Optional[Tuple[int, ...]] = None):
        self.origin_axes = axes

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        axes = [i for i in range(len(a.shape))]
        if self.origin_axes is None:
          axes[-2], axes[-1] = axes[-1], axes[-2]
        else:
          axes[self.origin_axes[1]], axes[self.origin_axes[0]] = axes[self.origin_axes[0]], axes[self.origin_axes[1]]
        return a.permute(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.origin_axes)
        ### END YOUR SOLUTION


def transpose(a: Tensor, axes: Optional[Tuple[int, ...]]=None) -> Tensor:
    return Transpose(axes)(a)


class Permute(TensorOp):
    def __init__(self, axes: Tuple[int, ...]):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return a.permute(self.axes)
    

def permute(a: Tensor, axes: Tuple[int, ...]) -> Tensor:
    return Permute(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return broadcast_inverse_op(a.shape, out_grad.shape, out_grad)
        ### END YOUR SOLUTION


def broadcast_to(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    return BroadcastTo(shape)(a)


class Maximum(TensorOp):
    def __init__(self, axes: Optional[Tuple[int, ...] | int] = None, keepdims = False):
        '''
        axes can be `int` type, because the NDArray.sum() only support single dimension summation
        '''
        if isinstance(axes, int):
            self.axes = (axes, )
        else:
            self.axes = axes
        self.keepdims = keepdims

    def compute(self, a: NDArray) -> NDArray:
        if self.axes is None or len(self.axes) == 1:
            return a.max(axis=self.axes, keepdims=self.keepdims)
        if self.keepdims:
            for axis in self.axes:
                a = a.max(axis=axis, keepdims=self.keepdims)
            return a
        axes = tuple(sorted(self.axes))
        reduced = 0
        for axis in axes:
            axis -= reduced
            a = a.max(axis=axis, keepdims=self.keepdims)
            reduced += 1
        return a
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        raise NotImplementedError()


def maximum(a: Tensor, axes: Optional[Tuple[int, ...] | int]=None, keepdims=False) -> Tensor:
    return Maximum(axes, keepdims)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[Tuple[int, ...] | int] = None, keepdims = False):
        '''
        axes can be `int` type, because the NDArray.sum() only support single dimension summation
        '''
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes
        self.keepdims = keepdims

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # attention that the `sum()` in NDArray is not support multiple dimensions
        if self.axes is None or len(self.axes) == 1:
            return a.sum(axis=self.axes, keepdims=self.keepdims)
        if self.keepdims:
            for axis in self.axes:
                a = a.sum(axis=axis, keepdims=self.keepdims)
            return a
        axes = tuple(sorted(self.axes))
        reduced = 0
        for axis in axes:
            axis -= reduced
            a = a.sum(axis=axis, keepdims=self.keepdims)
            reduced += 1
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if self.keepdims:
            result = broadcast_to(out_grad, a.shape)
        elif self.axes is None:
            shape = tuple(1 for _ in range(len(a.shape)))
            result = broadcast_to(reshape(out_grad, shape), a.shape)
        else:
            shape = list(out_grad.shape)
            for axis in self.axes:
                shape.insert(axis, 1)
            result = broadcast_to(reshape(out_grad, shape), a.shape)
        validate_grad(self, out_grad, node, result)
        return result
        ### END YOUR SOLUTION


def summation(a, axes: Optional[Tuple[int, ...] | int]=None, keepdims=False) -> Tensor:
    return Summation(axes, keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        a_bar, b_bar = matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad)
        if a_bar.shape != a.shape:
          expand = tuple(i for i in range(len(a_bar.shape) - len(a.shape)))
          a_bar = summation(a_bar, expand)
        if b_bar.shape != b.shape:
          expand = tuple(i for i in range(len(b_bar.shape) - len(b.shape)))
          b_bar = summation(b_bar, expand)
        return a_bar, b_bar
        ### END YOUR SOLUTION


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a: Tensor) -> Tensor:
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a: Tensor) -> Tensor:
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a: Tensor) -> Tensor:
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        mask = a > 0
        return a * mask
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        mask = a.cached_data > 0
        return out_grad * Tensor(mask, device=a.device, dtype=a.dtype)
        ### END YOUR SOLUTION


def relu(a: Tensor) -> Tensor:
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / ((exp(a) + exp(-a)) / 2) ** 2
        ### END YOUR SOLUTION


def tanh(a: Tensor) -> Tensor:
    return Tanh()(a)


class Unsqueeze(TensorOp):
    def __init__(self, dim: int):
        """
        Insert a dimension of size 1 at the specified position.
        
        Args:
            dim: Position where to insert the new dimension. Can be negative.
        """
        self.dim = dim

    def compute(self, a: NDArray) -> NDArray:
        shape = list(a.shape)
        # Handle negative dim
        if self.dim < 0:
            dim = len(shape) + self.dim + 1
        else:
            dim = self.dim
        # Insert dimension of size 1
        shape.insert(dim, 1)
        return a.compact().reshape(tuple(shape))

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return squeeze(out_grad, dim=self.dim)


def unsqueeze(a: Tensor, dim: int) -> Tensor:
    return Unsqueeze(dim)(a)


class Squeeze(TensorOp):
    def __init__(self, dim: Optional[int] = None):
        """
        Remove dimensions of size 1.
        
        Args:
            dim: If specified, only remove this dimension if it has size 1.
                 If None, remove all dimensions of size 1.
                 Can be negative.
        """
        self.dim = dim

    def compute(self, a: NDArray) -> NDArray:
        shape = list(a.shape)
        if self.dim is not None:
            # Handle negative dim
            if self.dim < 0:
                dim = len(shape) + self.dim
            else:
                dim = self.dim
            # Only remove the specified dimension if it has size 1
            if dim < len(shape) and shape[dim] == 1:
                shape.pop(dim)
        else:
            # Remove all dimensions of size 1
            shape = [s for s in shape if s != 1]
        return a.compact().reshape(tuple(shape))

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        original_shape = a.shape
        
        if self.dim is not None:
            # Only one dimension was removed, restore it using unsqueeze
            if self.dim < 0:
                dim = len(original_shape) + self.dim
            else:
                dim = self.dim
            return unsqueeze(out_grad, dim=dim)
        else:
            # Multiple dimensions might have been removed
            # Restore original shape by inserting 1s where they were removed
            out_shape = list(out_grad.shape)
            original_idx = 0
            out_idx = 0
            result_shape = []
            while original_idx < len(original_shape):
                if original_shape[original_idx] == 1:
                    # This dimension was removed, need to add it back
                    result_shape.append(1)
                    original_idx += 1
                else:
                    # This dimension was kept
                    if out_idx < len(out_shape):
                        result_shape.append(out_shape[out_idx])
                        out_idx += 1
                    original_idx += 1
            # Handle remaining dimensions in out_shape
            while out_idx < len(out_shape):
                result_shape.append(out_shape[out_idx])
                out_idx += 1
            return reshape(out_grad, tuple(result_shape))


def squeeze(a: Tensor, dim: Optional[int] = None) -> Tensor:
    return Squeeze(dim)(a)


class AdvancedIndexing(TensorOp):
    """
    Advanced indexing with integer tensors.
    Supports indexing like tensor[indices] where indices is an integer tensor.
    This operation uses numpy's advanced indexing, which requires data copying.
    
    Takes two inputs: (a, indices) where:
    - a: the tensor to index
    - indices: integer tensor used for indexing
    """
    def compute(self, a: NDArray, indices: NDArray) -> NDArray:
        # Get numpy arrays
        a_np = a.numpy()
        indices_np = indices.numpy()
        
        # Ensure indices are integers
        indices_np = indices_np.astype(numpy.int64)
        
        # Use numpy advanced indexing
        # This creates a copy of the data
        result_np = a_np[indices_np]
        
        # Create new NDArray from result
        return NDArray(result_np, device=a.device)

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Gradient for advanced indexing.
        The gradient is zero everywhere except at the indexed positions.
        We use numpy's advanced indexing to scatter gradients back.
        """
        a, indices = node.inputs
        
        # Create zero tensor with same shape as input
        grad_np = numpy.zeros(a.shape, dtype=a.dtype)
        
        # Get numpy arrays
        indices_np = indices.numpy()
        indices_np = indices_np.astype(numpy.int64)
        out_grad_np = out_grad.numpy()
        
        # Scatter gradients back to original positions
        # numpy's advanced indexing can handle this
        grad_np[indices_np] += out_grad_np
        
        # Indices don't have gradient (they're integer constants)
        return Tensor(grad_np, device=a.device, dtype=a.dtype), None


def advanced_indexing(a: Tensor, indices: Tensor) -> Tensor:
    return AdvancedIndexing()(a, indices)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: Tuple[NDArray, ...]) -> NDArray:
        ### BEGIN YOUR SOLUTION
        shape = list(args[0].shape)
        indices = [slice(0, shape[i], 1) for i in range(len(shape))]
        shape.insert(self.axis, len(args))
        indices.insert(self.axis, slice(0,1,1))
        result = args[0].device.full(shape=shape, fill_value=0)
        for i in range(len(args)):
          indices[self.axis] = slice(i, i+1, 1)
          result[tuple(indices)] = args[i]
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis: int) -> Tensor:
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A: NDArray) -> Tuple[NDArray, ...]:
        ### BEGIN YOUR SOLUTION
        indices = [slice(0, A.shape[i], 1) for i in range(len(A.shape))]
        shape = list(A.shape)
        shape.pop(self.axis)
        result = []
        for i in range(A.shape[self.axis]):
            indices[self.axis] = slice(i, i+1, 1)
            result.append(A[tuple(indices)].compact().reshape(tuple(shape)))
        return tuple(result)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: TensorTuple, node: Tensor):
        ### BEGIN YOUR SOLUTION
        assert isinstance(out_grad, TensorTuple)
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a: Tensor, axis: int) -> TensorTuple:
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Tuple[int, ...]):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a: Tensor, axes: Tuple[int, ...]) -> Tensor:
    return Flip(axes)(a)


class Pad(TensorOp):
    def __init__(self, axes: Tuple[int, ...]):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return a.pad(self.axes)
    

def pad(a: Tensor, axes: Tuple[int, ...]) -> Tensor:
    return Pad(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: Tuple[int, ...], dilation: int, full=True):
        self.axes = axes
        self.dilation = dilation
        self.full = full

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        new_shape= list(a.shape)
        origin_region_indices = [slice(0, dim, 1) for dim in a.shape]
        for axis in self.axes:
            if self.full:
                new_shape[axis] *= (self.dilation + 1)
            else:
                new_shape[axis] += self.dilation * (new_shape[axis]-1)
            origin_region_indices[axis] = slice(0, new_shape[axis], self.dilation + 1)
        result = NDArray.make(shape=tuple(new_shape), device=a.device)
        result[tuple(slice(0, dim, 1) for dim in new_shape)] = 0
        result[tuple(origin_region_indices)] = a
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def dilate(a: Tensor, axes: Tuple[int, ...], dilation: int, full=True) -> Tensor:
    return Dilate(axes, dilation, full=full)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: Tuple[int, ...], dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        remain_region_indices = [slice(0, dim, 1) for dim in a.shape]
        for axis in self.axes:
            new_shape[axis] //= (self.dilation + 1)
            remain_region_indices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        result = NDArray.make(shape=tuple(new_shape), device=a.device)
        result[tuple(slice(0, dim, 1) for dim in new_shape)] = a[tuple(remain_region_indices)]
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def undilate(a: Tensor, axes: Tuple[int, ...], dilation: int) -> Tensor:
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, Z: NDArray, W: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if self.padding != 0:
            Z = Z.pad(((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)))
        n, h, w, ic = Z.shape
        k, _, ic, oc = W.shape
        ns, hs, ws, cs = Z.strides
        nh, nw = (h-k)//self.stride+1, (w-k)//self.stride+1
        im2col = Z.as_strided(shape=(n, nh, nw, k, k, ic), strides=(ns, self.stride*hs, self.stride*ws, hs, ws, cs))
        inner_dim = k * k * ic
        output = im2col.compact().reshape((n*nh*nw, inner_dim)) @ W.compact().reshape((inner_dim, oc))
        return output.compact().reshape((n, nh, nw, oc))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z, W = node.inputs
        _, height, weight, _ = Z.shape
        K1, K2, _, _ = W.shape
        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride-1, full=False)
        # Z_grad
        h_padding, w_padding = K1-self.padding-1, K2-self.padding-1
        h_remain, w_remain = (height+2*self.padding-K1)%self.stride, (weight+2*self.padding-K2)%self.stride # remain region which kernel can not reach
        global_logger.debug(f'out_grad.shape: {out_grad.shape}, padding: ({h_padding},{h_padding+h_remain}),({w_padding},{w_padding+w_remain})')
        out_grad = pad(out_grad, ((0,0),(h_padding,h_padding+h_remain),(w_padding,w_padding+w_remain),(0,0)))
        flip_W = flip(permute(W, (0, 1, 3, 2)), (0, 1))
        Z_grad = conv(out_grad, flip_W)
        # W_grad
        Z_ = permute(Z, (3, 1, 2, 0)) # (IC, H, W, N) => (N', H, W, IC')
        Z_ = pad(Z_, ((0,0),(K1-1,K1-1),(K2-1,K2-1),(0,0)))
        out_grad_ = permute(out_grad, (1, 2, 0, 3)) # (NH, NW, N, OC) => (K1', K2', IC', OC)
        global_logger.debug(f'Z_.shape: {Z_.shape} out_grad_.shape: {out_grad_.shape}')
        W_grad = permute(conv(Z_, out_grad_), (1, 2, 0, 3))
        return Z_grad, W_grad
        ### END YOUR SOLUTION


def conv(Z: Tensor, W: Tensor, stride: Optional[int] = 1, padding: Optional[int] = 0):
    return Conv(stride, padding)(Z, W)


