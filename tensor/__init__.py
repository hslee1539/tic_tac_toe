from . import element_wise_product
from . import matrix_product
from . import transpose as transpose_module
from . import axis_module
from . import tool

from .class_tensor import *

def create_element_wise_product(left, right, dtype = float, stype = int):
    array, shape = element_wise_product.create_variables(left.array, left.shape, right.array, right.shape)
    return Tensor(array,shape)

def create_matrix_product(left, right, dtype = float, stype = int):
    array, shape = matrix_product.create_variables(left.array, left.shape, right.array, right.shape,dtype, stype)
    return Tensor(array, shape)

def create_sum(tensor, axis):
    array, shape = axis_module.create_variables(tensor.array, tensor.shape, axis)
    return Tensor(array, shape)

def create_transpose(tensor):
    array, shape = transpose_module.create_variables(tensor.array, tensor.shape)
    return Tensor(array, shape)

def add(left, right, out):
    element_wise_product.add(left.array, right.array, out.array)
    return out

def sub(left, right, out):
    element_wise_product.subtract(left.array, right.array, out.array)
    return out

def mul(left, right, out):
    element_wise_product.multiply(left.array, right.array, out.array)
    return out

def div(left, right, out):
    element_wise_product.divide(left.array, right.array, out.array)
    return out

def power(left, right, out):
    element_wise_product.power(left.array, right.array, out.array)
    return out

def matmul(left, right, out):
    matrix_product.matmul(left.array, left.shape, right.array, right.shape, out.array)
    return out

def argmax(tensor, axis, out):
    axis_module.argmax(tensor.array, tensor.shape, axis, out.array)
    return out

def sum_axis(tensor, axis, out):
    axis_module.sum_n(tensor.array, tensor.shape, axis, out.array)
    return out

def mul_axis(tensor, axis, out):
    axis_module.mul_n(tensor.array, tensor.shape, axis, out.array)
    return out

def mean_axis(tensor, axis, out):
    axis_module.mean_n(tensor.array, tensor.shape, axis, out.array)
    return out

def function(tensor, func, out):
    for i in range(len(out.array)):
        out.array[i] = func(tensor.array[i])
    return out

def function_elment_wise(left, right, func, out):
    element_wise_product.function(left.array, right.array, func,out.array)
    return out

def transpose(tensor, out):
    transpose_module.transpose(tensor.array, tensor.shape, out.array)
    return out

def set_randomly(max_range, out):
    tool.set_randomly(max_range, out.array)
    return out

def copy(source, start, length, out):
    tool.copy(source.array, start, length, out.array)
    return out

def copy_row(source, point, out):
    tool.copy_row(source.array, source.shape, point.array, out.array)
    return out

def set_shuffle(out):
    tool.set_shuffle(out.array)
    return out
