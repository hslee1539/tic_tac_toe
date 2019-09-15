def create_variables(array, shape):
    shape_len = len(shape)
    out_shape = shape.copy()
    for d in range(shape_len):
        out_shape[d] = shape[-1 -d % shape_len]
    return (array.copy(), out_shape)


def transpose(array, shape, out):
    """N-Tensor의 transpose연산만 합니다."""
    array_len = len(array)
    shape_len = len(shape)

    for i in range(array_len):
        index = i
        point = 0
        divider = array_len
        multiplier = 1
        for d in range(shape_len):
            divider = divider // shape[d]
            point = point + index // divider * multiplier
            index = index - index // divider * divider
            multiplier = multiplier * shape[-1 -(shape_len - 1 - d) % shape_len]
        out[point] = array[i]
    return out

