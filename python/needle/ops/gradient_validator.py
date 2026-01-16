import torch
import torch.autograd as autograd
import numpy as np
from typing import Dict, Callable, Optional, Tuple, Any
from ..autograd import Tensor, NDArray

class GradientValidator:
    """独立的梯度验证工具"""
    
    def __init__(self, enabled: bool = True, tolerance: float = 1e-5):
        self.enabled = enabled
        self.tolerance = tolerance
        self.torch_forward_registry: Dict[str, Callable] = {}
        self._register_default_ops()
    
    def register_torch_forward(self, op_name: str, torch_fn: Callable):
        """注册某个op的torch forward实现"""
        self.torch_forward_registry[op_name] = torch_fn
    
    def _register_default_ops(self):
        """注册默认的torch forward实现"""
        self.register_torch_forward('AddScalar', 
            lambda op, *inputs: inputs[0] + op.scalar)
        self.register_torch_forward('MulScalar',
            lambda op, *inputs: inputs[0] * op.scalar)
        self.register_torch_forward('Summation',
            lambda op, *inputs: torch.sum(inputs[0], dim=op.axes, keepdim=op.keepdims))
        self.register_torch_forward('EWiseDiv',
            lambda op, *inputs: inputs[0] / inputs[1])
        self.register_torch_forward('PowerScalar',
            lambda op, *inputs: torch.pow(inputs[0], op.scalar))
        self.register_torch_forward('EWiseMul',
            lambda op, *inputs: inputs[0] * inputs[1])

    def validate(self, op_instance, out_grad: Tensor, node: Tensor, 
                 actual_grad_result: Any) -> bool:
        """验证gradient实现"""
        if not self.enabled:
            return True
        
        op_name = op_instance.__class__.__name__
        print(op_name)
        if op_name not in self.torch_forward_registry:
            return True  # 未注册的op跳过验证
        
        print(f"{op_name}: registered!")
        # 获取forward输入
        inputs = [inp.realize_cached_data() for inp in node.inputs]
        torch_inputs = [torch.tensor(inp.numpy(), requires_grad=True) 
                       for inp in inputs]
        
        # 执行torch forward
        torch_fn = self.torch_forward_registry[op_name]
        torch_res = torch_fn(op_instance, *torch_inputs)
        
        # 计算torch gradient
        torch_out_grad = torch.tensor(out_grad.numpy())
        torch_grads = autograd.grad(
            torch_res, torch_inputs,
            grad_outputs=torch_out_grad,
            retain_graph=True
        )
        print(f'torch_grads: {torch_grads}')
        
        # 比较结果
        try:
            if isinstance(actual_grad_result, tuple):
                for i, (ndl_grad, torch_grad) in enumerate(
                    zip(actual_grad_result, torch_grads)):
                    if ndl_grad is not None and torch_grad is not None:
                        np.testing.assert_allclose(
                            ndl_grad.numpy(),
                            torch_grad.detach().numpy(),
                            rtol=self.tolerance,
                            atol=self.tolerance
                        )
            else:
                if actual_grad_result is not None and torch_grads[0] is not None:
                    np.testing.assert_allclose(
                        actual_grad_result.numpy(),
                        torch_grads[0].detach().numpy(),
                        rtol=self.tolerance,
                        atol=self.tolerance
                    )
            return True
        except AssertionError as e:
            print(f"❌ Gradient validation failed for {op_name}: {e}")
            return False


# 全局验证器实例
_global_validator = GradientValidator(enabled=True)  # 默认关闭

def enable_gradient_validation(enabled: bool = True, tolerance: float = 1e-5):
    """启用/禁用梯度验证"""
    global _global_validator
    _global_validator.enabled = enabled
    _global_validator.tolerance = tolerance

def get_validator() -> GradientValidator:
    """获取全局验证器"""
    return _global_validator