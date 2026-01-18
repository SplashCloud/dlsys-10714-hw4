import operator
import math
from functools import reduce
import numpy as np
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu


# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return cpu_numpy()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    """A generic ND array class that may contain multipe different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.

    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape):
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### Properies and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    def __repr__(self):
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value):
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device):
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self):
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self):
        """Convert a matrix to be compact"""
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape, strides):
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle, offset=self._offset
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            NDArray : reshaped array; this will point to thep
        """

        ### BEGIN YOUR SOLUTION
        if not self.is_compact():
          raise ValueError("The current matrix is not compact")
        if len(new_shape) == 2: 
            # check if -1 exists and convert -1 to corresponding value
            new_shape = list(new_shape)
            if new_shape[0] == -1:
                new_shape[0] = prod(self._shape) // new_shape[1]
            elif new_shape[1] == -1:
                new_shape[1] = prod(self._shape) // new_shape[0]
            new_shape = tuple(new_shape)
        if prod(new_shape) != prod(self._shape):
          raise ValueError("The new size != current size")
        new_strides = self.compact_strides(new_shape)
        return self.make(shape=new_shape, strides=new_strides, device=self._device, handle=self._handle, offset=self._offset)
        ### END YOUR SOLUTION

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permuation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memroy as the original array.

        Args:
            new_axes (tuple): permuation order of the dimensions

        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """

        ### BEGIN YOUR SOLUTION
        new_shape = tuple(np.array(self._shape)[np.array(new_axes)])
        new_strides = tuple(np.array(self._strides)[np.array(new_axes)]) # strides也要跟着改变顺序，因为获取元素的时候实际是strides在决定
        return self.make(shape=new_shape, strides=new_strides, device=self._device, handle=self._handle, offset=self._offset)
        ### END YOUR SOLUTION

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """

        ### BEGIN YOUR SOLUTION
        new_strides = []
        for old, new, stride in zip(reversed(self._shape), reversed(new_shape), reversed(self._strides)):
          if old != 1 and old != new:
            raise ValueError(f"old({old}) != new({new})")
          elif old == new:
            new_strides.append(stride)
          else:
            new_strides.append(0)
        new_strides.extend([0 for _ in range(len(new_shape) - len(self._shape))])
        return self.make(shape=new_shape, strides=tuple(reversed(new_strides)), device=self._device, handle=self._handle, offset=self._offset)
        ### END YOUR SOLUTION

    ### Get and set elements

    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start += self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop += self.shape[dim]
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.

        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            coresponding to the subset of the matrix to get

        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memroy but just
            manipulate the shape/strides/offset of the new array, referecing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        if len(idxs) != self.ndim:
            # if len(idx) < ndim, padding with (0:end:1)
            for i in range(self.ndim):
                if i < len(idxs):
                    continue
                idxs += (slice(0, self.shape[i], 1),)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        ### BEGIN YOUR SOLUTION
        new_shape = []
        new_strides = [] # strides also need to change when (step != 1)
        new_offset = 0
        for i, s in enumerate(idxs):
          start, stop, step = s.start, s.stop, s.step
          new_shape.append((stop - start - 1) // step + 1)
          new_strides.append(self._strides[i] * step)
          new_offset += self._strides[i] * start
        return self.make(shape=tuple(new_shape), strides=tuple(new_strides), device=self._device, handle=self._handle, offset=self._offset+new_offset)
        ### END YOUR SOLUTION

    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    @staticmethod
    def broadcast_shape(shape1, shape2):
        """Compute the broadcast shape of two shapes following numpy broadcasting rules."""
        # Pad the shorter shape with 1s from the left
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)
        
        # Pad shapes to the same length
        padded_shape1 = (1,) * (max_len - len1) + shape1
        padded_shape2 = (1,) * (max_len - len2) + shape2
        
        # Compute broadcast shape
        broadcast_shape = []
        for s1, s2 in zip(padded_shape1, padded_shape2):
            if s1 == s2:
                broadcast_shape.append(s1)
            elif s1 == 1:
                broadcast_shape.append(s2)
            elif s2 == 1:
                broadcast_shape.append(s1)
            else:
                raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}: incompatible dimensions")
        
        return tuple(broadcast_shape)

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar.
        Supports broadcasting when both operands are NDArrays.
        """
        if isinstance(other, NDArray):
            # Compute broadcast shape
            out_shape = self.broadcast_shape(self.shape, other.shape)
            
            # Broadcast both arrays to the output shape
            self_broadcast = self.broadcast_to(out_shape) if self.shape != out_shape else self
            other_broadcast = other.broadcast_to(out_shape) if other.shape != out_shape else other
            
            # Create output array with broadcast shape
            out = NDArray.make(out_shape, device=self.device)
            
            # Perform elementwise operation on broadcasted arrays
            ewise_func(self_broadcast.compact()._handle, other_broadcast.compact()._handle, out._handle)
        else:
            out = NDArray.make(self.shape, device=self.device)
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):
        '''
        extend to multi-dimensions matrix mulitplication
        '''
        if self.ndim == 2 and other.ndim == 2:
            return self._matmul_2d(self, other)
        
        a = self
        b: NDArray = other

        if a.ndim == 1:
            a = a.compact().reshape((1, a.shape[0]))
        if b.ndim == 1:
            b = b.compact().reshape((b.shape[0], 1))

        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("MM requires at least 2D arrays")

        if a.shape[-1] != b.shape[-2]:
            raise ValueError("Incompatible shapes for matmul: {a.shape} and {b.shape}")

        m, k = a.shape[-2], a.shape[-1]
        k2, n = b.shape[-2], b.shape[-1]
        assert k == k2

        other_dims_a = a.shape[:-2]
        other_dims_b = b.shape[:-2]

        output_batch_dims = self.broadcast_shape(other_dims_a, other_dims_b)

        if other_dims_a != output_batch_dims:
            a = a.broadcast_to(output_batch_dims + (m, k))
        if other_dims_b != output_batch_dims:
            b = b.broadcast_to(output_batch_dims + (k, n))

        batch_size = prod(output_batch_dims) if output_batch_dims else 1

        if batch_size == 1:
            a_2d = a.compact().reshape((m, k))
            b_2d = b.compact().reshape((k, n))
            result_2d = self._matmul_2d(a_2d, b_2d)
            return result_2d.compact().reshape(output_batch_dims + (m, n))

        a_flat = a.compact().reshape((batch_size, m, k))
        b_flat = b.compact().reshape((batch_size, k, n))
        out_flat = NDArray.make((batch_size, m, n), device=self.device)

        for i in range(batch_size):
            a_i = a_flat[i].compact().reshape((m, k))
            b_i = b_flat[i].compact().reshape((k, n))
            out_i = self._matmul_2d(a_i, b_i)
            out_flat[i] = out_i.compact().reshape((m, n))

        return out_flat.compact().reshape(output_batch_dims + (m, n))
        

    def _matmul_2d(self, a, b):
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will restride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """
        assert isinstance(a, NDArray) and isinstance(b, NDArray)
        assert a.ndim == 2 and b.ndim == 2
        assert a.shape[1] == b.shape[0]

        m, n, p = a.shape[0], a.shape[1], b.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(a.device, "matmul_tiled") and all(
            d % a.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = a.device.__tile_size__
            a = tile(a.compact(), t).compact()
            b = tile(b.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=a.device)
            a.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((a.shape[0], b.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=a.device)
            a.device.matmul(
                a.compact()._handle, b.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis, keepdims=False):
        """ Return a view to the array set up for reduction functions and output array. """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            #out = NDArray.make((1,) * self.ndim, device=self.device)
            # out = NDArray.make((1,), device=self.device)
            out = NDArray.make(
                tuple(1 for _ in range(len(self.shape)))
                if keepdims else
                (),  # Return scalar (0-d array) when keepdims=False and axis is None(means reduce over all axes)
                device=self.device
            )

        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            # Normalize negative axis to positive axis
            if axis < 0:
                axis = axis + self.ndim
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"Axis {axis} is out of bounds for array of dimension {self.ndim}")

            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),
                device=self.device,
            )
        return view, out

    def sum(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def flip(self, axes):
        """
        Flip this ndarray along the specified axes.
        Note: compact() before returning.
        """
        ### BEGIN YOUR SOLUTION
        new_strides = list(self.strides)
        new_offset = 0
        for axis in axes:
            new_strides[axis] *= -1
            new_offset += (self.shape[axis] - 1) * self.strides[axis]
        result = NDArray.make(shape=self._shape, strides=new_strides, device=self._device, handle=self._handle,  offset=new_offset)
        return result.compact()
        ### END YOUR SOLUTION

    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
        """
        ### BEGIN YOUR SOLUTION
        new_shape = []
        origin_region_indices = []
        for pad, dim in zip(axes, self.shape):
            new_shape.append(dim + pad[0] + pad[1])
            origin_region_indices.append(slice(pad[0], pad[0]+dim, 1))
        result = NDArray.make(shape=tuple(new_shape), device=self.device)
        result[tuple(slice(0, dim, 1) for dim in new_shape)] = 0
        result[tuple(origin_region_indices)] = self
        return result
        ### END YOUR SOLUTION

def array(a, dtype="float32", device=None):
    """Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)


def reshape(array, new_shape):
    return array.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()


def sum(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)


def flip(a, axes):
    return a.flip(axes)
