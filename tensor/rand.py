from . import class_tensor as _tensor
import random as _random

_rd = _random.Random()

def create_array(length):
    tmp = [None] * length
    for i in range(length):
        tmp[i] = _rd.random()
    return tmp

def create_tensor(shape):
    array_len = 1
    for i in range(len(shape)):
        array_len *= shape[i]
    return _tensor.Tensor(create_array(array_len), shape)
