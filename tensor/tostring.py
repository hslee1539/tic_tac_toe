def tostring(array, shape, newline = "\n"):
    out = [""]
    #todo: 배열 -> 정수로 
    pos = [0] * len(shape)

    #do-while
    _tostring_process1(out, pos)
    _tostring_process2(out, array, shape, pos)
    _tostring_process3(out, shape, pos)
    while(_tostring_process4(out, shape, pos, newline)):
        _tostring_process1(out, pos)
        _tostring_process2(out, array, shape, pos)
        _tostring_process3(out, shape, pos)
    return str(len(shape)) + '-Tensor' + newline + out[0]

def _tostring_process1(out, pos):
    count = 0
    for item in pos:
        if(item > 0):
            count = 0
        else:
            count += 1
    for i in range(len(pos) - count):
        out[0] += ' '
    for i in range(count):
        out[0] += '['
    return None

def _tostring_process2(out, array, shape, pos):
    point = 0
    multipler = 1
    i = len(shape) - 1
    while(i > -1):
        point += multipler * (pos[i] % shape[i])
        multipler *= shape[i]
        i -= 1
        
    out[0] += str(array[point])
    if(len(shape) > 0):
        for i in range(1, shape[-1]):
            out[0] += ', ' + str(array[point + i])
        pos[len(shape) - 1] = shape[-1]
        _tostring_process2_stack(shape,pos,1)
    return None

def _tostring_process2_stack(shape, pos, i):
    pos[-i] = 0
    if(len(shape) > i ):
        pos[-i-1] += 1
        if(pos[-i -1] == shape[-i -1]):
            _tostring_process2_stack(shape, pos, i + 1)
    return None

def _tostring_process3(out, shape, pos):
    count = 0
    for item in pos:
        if (item > 0):
            count = 0
        else:
            count += 1
    for i in range(count):
        out[0] += ']'
    return None

def _tostring_process4(out, shape, pos, newline):
    count = 0
    out[0] += newline
    for item in pos:
        if(item == 0):
            count += 1
    if (len(shape) == count):
        return False
    else:
        return True
