def create_variables(array, shape, axis):
    shape_len = len(shape)
    axis = (shape_len + axis) % shape_len
    tmp_shape = [type(shape[0])(0)] * (shape_len - 1)
    for i in range(axis):
        tmp_shape[i] = shape[i]
    for i in range(axis + 1, shape_len):
        tmp_shape[i - 1] = shape[i]
    tmp_array = [type(array[0])(0)] * (len(array) // shape[axis])
    return tmp_array, tmp_shape

def argmax(array, shape, axis, out):
    shape_len = len(shape)
    axis = (shape_len + axis) % shape_len
    axis_mover2 = 1
    axis_value = shape[axis]

    
    for i in range(axis + 1, shape_len):
        axis_mover2 *= shape[i]
    axis_mover1 = (axis_mover2 * axis_value) * (1 % (axis + 1)) + 1 -1 % (axis + 1)

    max_val = 0
    tmp = 0
    
    for i in range(len(out)):
        out[i] = 0
        max_val = array[i%axis_mover2 + i//axis_mover2*axis_mover1]
        for j in range(shape[axis]):
            tmp = array[i%axis_mover2 + i//axis_mover2*axis_mover1 + axis_mover2 * j]
            if(max_val < tmp):
                max_val = tmp
                out[i] = j
    return out

def sum_n(array, shape, axis, out):
    shape_len = len(shape)
    axis = (shape_len + axis) % shape_len
    axis_mover2 = 1
    axis_value = shape[axis]

    for i in range(axis + 1, shape_len):
        axis_mover2 *= shape[i]
    axis_mover1 = (axis_mover2 * axis_value) * (1 % (axis + 1)) + 1 -1 % (axis + 1)
    
    for i in range(len(out)):
        out[i] = 0
        for j in range(shape[axis]):
            out[i] += array[i%axis_mover2 + i//axis_mover2*axis_mover1 + axis_mover2 * j]
    return out

def mul_n(array, shape, axis, out):
    shape_len = len(shape)
    axis = (shape_len + axis) % shape_len
    axis_mover2 = 1
    axis_value = shape[axis]

    
    for i in range(axis + 1, shape_len):
        axis_mover2 *= shape[i]
    axis_mover1 = (axis_mover2 * axis_value) * (1 % (axis + 1)) + 1 -1 % (axis + 1)
    
    for i in range(len(out)):
        out[i] = 1
        for j in range(shape[axis]):
            out[i] *= array[i%axis_mover2 + i//axis_mover2*axis_mover1 + axis_mover2 * j]
    return out

def mean_n(array, shape, axis, out):
    shape_len = len(shape)
    axis = (shape_len + axis) % shape_len
    axis_mover2 = 1
    axis_value = shape[axis]

    
    for i in range(axis + 1, shape_len):
        axis_mover2 *= shape[i]
    axis_mover1 = (axis_mover2 * axis_value) * (1 % (axis + 1)) + 1 -1 % (axis + 1)
    
    for i in range(len(out)):
        out[i] = 0
        for j in range(shape[axis]):
            out[i] += array[i%axis_mover2 + i//axis_mover2*axis_mover1 + axis_mover2 * j]
        out[i] /= shape[axis]
    return out

