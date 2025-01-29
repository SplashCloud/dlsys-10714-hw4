"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=self.in_channels*self.kernel_size*self.kernel_size,
            fan_out=self.out_channels*self.kernel_size*self.kernel_size,
            shape=(self.kernel_size, self.kernel_size, self.in_channels, self.out_channels),
            device=device,
            dtype=dtype,
            requires_grad=True
        ))
        self.bias = Parameter(init.rand(
            *(self.out_channels,),
            low=-1/math.sqrt(self.in_channels*self.kernel_size*self.kernel_size),
            high=1/math.sqrt(self.in_channels*self.kernel_size*self.kernel_size),
            device=device,
            dtype=dtype,
            requires_grad=True
        )) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_ = ops.transpose(ops.transpose(x, axes=(1,2)), axes=(2,3)) # x: NCHW -> NHWC
        _, H, W, _ = x_.shape
        assert H == W, "only support square input"
        pad = self.kernel_size//2 # (H+2*P-K)//S+1=(H+S-1)//S
        o = ops.conv(x_, self.weight, stride=self.stride, padding=pad)
        if self.bias is not None:
            o = o + ops.broadcast_to(self.bias, shape=o.shape)
        return ops.transpose(ops.transpose(o, axes=(2,3)), axes=(1,2)) # NHWC -> NCHW
        ### END YOUR SOLUTION