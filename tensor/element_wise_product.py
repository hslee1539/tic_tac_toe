def create_variables(left_array, left_shape, right_array, right_shape):
    if(len(left_array) > len(right_array)):
        return (left_array.copy(), left_shape.copy())
    else:
        return (right_array.copy(), right_shape.copy())

def add(left, right, out):
    """1차원 배열로 element wise product 덧셈 연산합니다."""
    left_len = len(left)
    right_len = len(right)
    for i in range(len(out)):
        out[i] = left[i % left_len] + right[i % right_len]
    return out

def subtract(left, right, out):
    left_len = len(left)
    right_len = len(right)
    for i in range(len(out)):
        out[i] = left[i % left_len] - right[i % right_len]
    return out

def multiply(left, right, out):
    """1차원 배열로 element wise product 곱셈 연산합니다."""
    left_len = len(left)
    right_len = len(right)
    for i in range(len(out)):
        out[i] = left[i % left_len] * right[i % right_len]
    return out

def divide(left, right, out):
    left_len = len(left)
    right_len = len(right)
    for i in range(len(out)):
        out[i] = left[i % left_len] / right[i % right_len]
    return out

def power(left, right, out):
    left_len = len(left)
    right_len = len(right)
    for i in range(len(out)):
        out[i] = left[i % left_len] ** right[i % right_len]
    return out

def function(left, right, func, out):
    left_len = len(left)
    right_len = len(right)
    for i in range(len(out)):
        out[i] = func(left[i % left_len], right[i % right_len])
    return out
