def matmul(left_array, left_shape, right_array, right_shape, out):
    left_len = len (left_array)
    right_len = len(right_array)
    left_shape_len = len(left_shape)
    right_shape_len = len(right_shape)
    left_sw = 2 // (left_shape_len + 1)
    right_sw = 2 // (right_shape_len + 1)
    row = left_shape[-1 -1 % left_shape_len] * (1 - left_sw) + left_sw
    col = right_shape[-1] * (1 - right_sw) + right_sw

    matrix2d = col * row

    right_matrix = col * right_shape[-1 -1 % right_shape_len]

    product_max = left_shape[-1]

    step_max = len(out)
    step = 0
    while(step < step_max):
        left_product_step = step // col * product_max
        right_product_step = step // matrix2d * right_matrix + step % col

        product_step = 0
        out[step] = 0
        while(product_step < product_max):
            out[step] += left_array[
                (left_product_step + product_step) % left_len
                ] * right_array[
                    (right_product_step + product_step * col) % right_len
                ]
            product_step += 1
        step += 1
    return None

def create_variables (left_array, left_shape, right_array, right_shape, dtype = float, stype = int):
    #step1: 최고 차원의 텐서를 찾기.
    #목적 : shape을 결정하는데, shape의 -1과 -2를 뺴면, 최고 차원의 shape과 같음. (단 left 또는 right 중, 1d가 있으면, 조금 달라짐.)
    left_len = len(left_array)
    right_len = len(right_array)
    left_shape_len = len(left_shape)
    right_shape_len = len(right_shape)
    if(left_shape_len == right_shape_len):
        if(left_len > right_len):
            higher_array = left_array
            higher_shape = left_shape
        else:
            higher_array = right_array
            higher_shape = right_shape
    elif(left_shape_len > right_shape_len):
        higher_array = left_array
        higher_shape = left_shape
    else:
        higher_array = right_array
        higher_shape = right_shape
    #장점:
    #0d 일 경우, 1 // 0이 되어, 계산 애러가 나옴.
    #numpy matmul도 1d일 경우, 애러를 표시하고, multiply를 대신 사용하라는 메세지가 나옴.
    is_left_1d = 1 // left_shape_len
    is_right_1d = 1 // right_shape_len
    #0d를 허용하고 싶을 경우 코드
    #left_sw = 2 // (len(left_shape) + 1)
    #right_sw = 2 // (len(right_shape) + 1)

    #shape의 길이는 left와 right의 최고 차원의 shape의 길이를 따르고, 1d가 있을 경우, 차원 감소함.
    out_shape_len = len(higher_shape) - (is_left_1d ^ is_right_1d)
    out_shape = [stype(0)] * out_shape_len

    #-3 이하의 차원의 모양은 최고 차원의 shape을 따르므로, 이를 복사함.
    for i in range(out_shape_len - 2):
        out_shape[i] = higher_shape[i]

    is_not_left_1d = 1 - is_left_1d
    is_not_right_1d = 1 - is_right_1d
    is_left_3d_or_more = 2 % left_shape_len // 2
    is_right_3d_or_more = 2 % right_shape_len // 2
    is_not_left_3d_or_more = 1 - is_left_3d_or_more
    is_not_right_3d_or_more = 1 - is_right_3d_or_more
    
    left_minus_1d = left_shape[-1]
    right_minus_1d = right_shape[-1]
    left_minus_2d = left_shape[-1 -1 % left_shape_len] # 만약 아니면, -1d를 가짐.
    right_minus_2d = right_shape[-1 -1 % right_shape_len] # 만약 아니면, -1d를 가짐.
    left_minus_3d = left_shape[-1 -2 % left_shape_len] # 만약 아니면, -1d를 가짐.
    right_minus_3d = right_shape[-1 -2 % right_shape_len]# 만약 아니면, -1d를 가짐.
    
    #-1 차원의 shape 결정
    out_shape[-1] = (
        right_minus_1d * is_not_left_1d * is_not_right_1d # 만약 모두 2d 이상인 경우
        + right_minus_1d * is_left_1d * is_not_right_1d # 만약 왼쪽이 1d인 경우
        + left_minus_2d * is_not_left_1d * is_right_1d)# 만약 오른쪽이 1d인 경우
        #+ is_left_1d * is_right_1d)#이 코드는 아래 코드와 중복이 됨.
    #-2 차원의 shape 결정.
    out_shape[-1 -1 % out_shape_len] = (
        left_minus_2d * is_not_left_1d * is_not_right_1d
        + right_minus_3d * is_left_1d * is_right_3d_or_more
        + left_minus_3d * is_left_3d_or_more * is_right_1d
        + out_shape[-1] * is_left_1d * is_not_right_1d * is_not_right_3d_or_more#만약 왼쪽은 1d, 오른쪽은 2d인 경우
        + out_shape[-1] * is_not_left_1d * is_right_1d * is_not_left_3d_or_more#만약 왼쪽은 2d, 오른쪽은 1d인 경우
        + is_left_1d * is_right_1d)#모두 1d인 경우. (이 경우, out_shape은 식에 의해 -1을 가리킴.)
    
    out_array = [dtype(0)] * ( (left_minus_2d * is_not_left_1d + is_left_1d)
                               * (right_minus_1d * is_not_right_1d + is_right_1d)
                               * len(higher_array)
                               // higher_shape[-1]
                               // higher_shape[-1 -1 % len(higher_shape)])
    return out_array, out_shape
