import numpy as np
import chainer
from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Sqrt(function_node.FunctionNode):
    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        self.retain_outputs((0,))
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.sqrt(x[0], dtype=x[0].dtype)),

    def backward(self, indexes, grad_outputs):
        y = self.get_retained_outputs()[0]
        gy = grad_outputs[0]
        return SqrtGrad().apply((y, gy))

class SqrtGrad(function_node.FunctionNode):
    def __init__(self):
        super(SqrtGrad, self).__init__()
        
    def forward(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        return utils.force_array(gy / (2 * y)),

    def backward(self, indexes, grad_outputs):
        y, gy = self.get_retained_inputs()
        g = grad_outputs[0]

        grad_gy = g / (2 * y)
        grad_y = - grad_gy * g / y
        return grad_y, grad_gy

def my_sqrt(x):
    return Sqrt().apply((x,))[0]

class RSqrt(function_node.FunctionNode):
    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        self.retain_outputs((0,))
        xp = cuda.get_array_module(*x)
        return utils.force_array(1 / xp.sqrt(x[0], dtype=x[0].dtype)),

    def backward(self, indexes, grad_outputs):
        y = self.get_retained_outputs()[0]
        gy = grad_outputs[0]
        return RSqrtGrad().apply((y, gy))

class RSqrtGrad(function_node.FunctionNode):
    def __init__(self):
        super(RSqrtGrad, self).__init__()
        
    def forward(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs
        return utils.force_array(- gy * (y ** 3) / 2),

    def backward(self, indexes, grad_outputs):
        y, gy = self.get_retained_inputs()
        g = grad_outputs[0]

        mgyyd2 = - g * (y ** 2) / 2
        grad_y = 3 * g * mgyyd2
        grad_gy = y * mgyyd2
        return grad_y, grad_gy

def my_rsqrt(x):
    return RSqrt().apply((x,))[0]
