from . import tostring as _tostring
import random as _random

_rd = _random.Random()

class Tensor:
    """N차원 텐서를 관리하는 클래스입니다."""
    def __init__(self, array, shape):
        """n-tensor를 만듭니다. array, shape은 참조 복사합니다."""
        self.array = array
        self.shape = shape

    def __str__(self):
        return _tostring.tostring(self.array, self.shape)

    def __repr__(self):
        return _tostring.tostring(self.array, self.shape)

    def copy(self):
        """이 객체를 깊은 복사로 복사합니다."""
        return Tensor(self.array.copy(), self.shape.copy())


def create_randomly(shape):
    array_len = 1
    for i in range(len(shape)):
        array_len *= shape[i]

    array = [None] * array_len
    for i in range(array_len):
        array[i] = _rd.random() * 2 - 1
    
    return Tensor(array, shape)

def create_zeros(shape, dtype = float):
    multipler = 1
    for i in range(len(shape)):
        multipler *= shape[i]
    return Tensor([dtype(0)] * multipler, shape)

def create_ones(shape, dtype = float):
    multipler = 1
    for i in range(len(shape)):
        multipler *= shape[i]
    return Tensor([dtype(1)] * multipler, shape)

def create_arange(start, end, step = 1):
    count = 0
    length = int((end - start) / step)
    array = [(end - start) / step] * length
    for i in range(length):
        array[i] = start + step * i
    return Tensor(array, [length])
