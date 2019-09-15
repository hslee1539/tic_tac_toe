import tensor
import numpy as np
import common.layers
import layer
import random
import time

rand = random.Random()

def parseNumpy(tensor):
    tmp = np.array(tensor.array, dtype = type(tensor.array[0]))
    return tmp.reshape(tensor.shape)
    
def parseTensor(numpy_array):
    return tensor.Tensor(numpy_array.reshape([-1]).tolist(), numpy_array.shape)

def compare(tensor, numpy_array, error = 0.001):
    tmp = parseNumpy(tensor)
    tmp = tmp - numpy_array
    tmp1 = (tmp < error) & (tmp > -error)
    tmp2 = tmp1.reshape([-1])
    tmp3 = True
    for i in range(tmp2.size):
        if(tmp2[i] == False):
            tmp3 = tmp1
    return tmp3

def test_matmul(shape1, shape2, error = 0.001):
    t1 = tensor.create_randomly(shape1)
    t2 = tensor.create_randomly(shape2)
    n1 = parseNumpy(t1)
    n2 = parseNumpy(t2)
    t = tensor.matmul(t1, t2, tensor.create_matrix_product(t1, t2))
    n = np.matmul(n1, n2)
    print(t)
    print(n)
    return compare(t, n)

def test_axis(shape, axis, error = 0.001):
    t1 = tensor.create_randomly(shape)
    n1 = parseNumpy(t1)
    t = tensor.sum_axis(t1, axis, tensor.create_sum(t1, axis))
    n = np.sum(n1, axis)
    print(t1)
    print(t)
    print(n)
    return compare(t,n)

def test_affine(data_shape, w_shape, error = 0.00001):
    tensor_data = tensor.create_randomly(data_shape)
    tensor_w = tensor.create_randomly(w_shape)
    tensor_b = tensor.create_randomly([w_shape[-1]])
    numpy_data = parseNumpy(tensor_data)
    numpy_w = parseNumpy(tensor_w)
    numpy_b = parseNumpy(tensor_b)
    tensor_layer = layer.Affine(tensor_w, tensor_b)
    numpy_layer = common.layers.Affine(numpy_w, numpy_b)

    tensor_y = tensor_layer.forward(tensor_data).copy()
    numpy_y = numpy_layer.forward(numpy_data)
    tensor_dout = tensor.create_randomly(tensor_y.shape)
    numpy_dout = parseNumpy(tensor_dout)

    tensor_dout = tensor_layer.backward(tensor_dout)
    numpy_dout = numpy_layer.backward(numpy_dout)

    print("forward")
    print(compare(tensor_y, numpy_y))
    print("backward")
    print(compare(tensor_dout, numpy_dout))
    print("dW")
    print(compare(tensor_layer.dW, numpy_layer.dW))
    print("db")
    print(compare(tensor_layer.db, numpy_layer.db))

def test_sigmoid(shape, error = 0.0001):
    tensor_data = tensor.create_randomly(shape)
    numpy_data = parseNumpy(tensor_data)

    tensor_layer = layer.Sigmoid()
    numpy_layer = common.layers.Sigmoid()

    tensor_y = tensor_layer.forward(tensor_data).copy()
    numpy_y = numpy_layer.forward(numpy_data)

    tensor_dout = tensor.create_randomly(tensor_y.shape)
    numpy_dout = parseNumpy(tensor_dout)

    tensor_dout = tensor_layer.backward(tensor_dout)
    numpy_dout = numpy_layer.backward(numpy_dout)

    print("forward")
    print(tensor_y)
    print(numpy_y)
    print(compare(tensor_y, numpy_y, error))
    print("backward")
    print(compare(tensor_dout, numpy_dout, error))

def test_relu(shape, error = 0.0001):
    tensor_data = tensor.create_randomly(shape)
    numpy_data = parseNumpy(tensor_data)

    tensor_layer = layer.Relu()
    numpy_layer = common.layers.Relu()

    tensor_y = tensor_layer.forward(tensor_data).copy()
    numpy_y = numpy_layer.forward(numpy_data)

    tensor_dout = tensor.create_randomly(tensor_y.shape)
    numpy_dout = parseNumpy(tensor_dout)

    tensor_dout = tensor_layer.backward(tensor_dout)
    numpy_dout = numpy_layer.backward(numpy_dout)

    print("forward")
    print(tensor_y)
    print(numpy_y)
    print(compare(tensor_y, numpy_y, error))
    print("backward")
    print(compare(tensor_dout, numpy_dout, error))

def test_softmax_old(y_shape, one_hot_table, error = 0.001):
    tensor_y = tensor.create_randomly(y_shape)
    numpy_y = parseNumpy(tensor_y)

    if(one_hot_table):        
        tensor_t = tensor.create_zeros(y_shape, int)
        
        for i in range(y_shape[0]):
            tensor_t.array[i * y_shape[-1] + rand.randint(0, y_shape[-1] - 1)] = 1
            
    else:
        tensor_t = tensor.create_zeros([y_shape[0], 1], int)
        for i in range(y_shape[0]):
            tensor_t.array[i] = rand.randint(0, y_shape[-1] - 1)

    print(type(tensor_t.array[0]))
    numpy_t = parseNumpy(tensor_t)
    print(numpy_t.dtype)

    tensor_layer = layer.SoftmaxWithLoss()
    numpy_layer = common.layers.SoftmaxWithLoss()

    print("loss")
    print(tensor_layer.forward(tensor_y, tensor_t))
    print(numpy_layer.forward(numpy_y, numpy_t))
    tensor_y = tensor_layer.out.copy()
    numpy_y = numpy_layer.y

    tensor_dout = tensor_layer.backward()
    numpy_dout = numpy_layer.backward()

    print("forward")
    print(tensor_y)
    print(numpy_y)
    print(compare(tensor_y, numpy_y, error))
    print("backward")
    print(compare(tensor_dout, numpy_dout, error))



def test_softmax(y_shape, one_hot_table, error = 0.001):
    tensor_y = tensor.create_randomly(y_shape)
    numpy_y = parseNumpy(tensor_y)

    if(one_hot_table):        
        tensor_t = tensor.create_zeros(y_shape, int)
        
        for i in range(y_shape[0]):
            tensor_t.array[i * y_shape[-1] + rand.randint(0, y_shape[-1] - 1)] = 1
            
    else:
        tensor_t = tensor.create_zeros([y_shape[0], 1], int)
        for i in range(y_shape[0]):
            tensor_t.array[i] = rand.randint(0, y_shape[-1] - 1)

    numpy_t = parseNumpy(tensor_t)
    print(numpy_t)

    tensor_layer = layer.Softmax()
    numpy_layer = common.layers.SoftmaxWithLoss()

    tensor_layer.forward(tensor_y)
    numpy_loss = numpy_layer.forward(numpy_y, numpy_t)

    tensor_y = tensor_layer.out.copy()
    numpy_y = numpy_layer.y

    tensor_layer.init_table(tensor_t)
    tensor_dout = tensor_layer.backward(1)
    numpy_dout = numpy_layer.backward()

    print("loss")
    print(tensor_layer.loss)
    print(numpy_loss)
    print("forward")
    print(tensor_y)
    print(numpy_y)
    print(compare(tensor_y, numpy_y, error))
    print("backward")
    print(tensor_dout)
    print(numpy_dout)
    print(compare(tensor_dout, numpy_dout, error))

def test_layer(data_shape, layer1_shape, layer2_shape, loop_count, error = 0.001):
    tensor_x = tensor.create_randomly(data_shape)
    numpy_x = parseNumpy(tensor_x)
    
    tensor_t = tensor.create_zeros([data_shape[0], layer2_shape[-1]], int)
    for i in range(data_shape[0]):
        tensor_t.array[i * layer2_shape[-1] + rand.randint(0, layer2_shape[-1] - 1)] = 1

    numpy_t = parseNumpy(tensor_t)

    tensor_w1 = tensor.create_randomly(layer1_shape)
    tensor_b1 = tensor.create_randomly([layer1_shape[-1]])

    tensor_w2 = tensor.create_randomly(layer2_shape)
    tensor_b2 = tensor.create_randomly([layer2_shape[-1]])

    numpy_w1 = parseNumpy(tensor_w1)
    numpy_b1 = parseNumpy(tensor_b1)

    numpy_w2 = parseNumpy(tensor_w2)
    numpy_b2 = parseNumpy(tensor_b2)


    #layer
    import layer
    tensor_layer = layer.Layers()
    tensor_layer.append_affine(tensor_w1, tensor_b1)
    tensor_layer.append_relu()
    tensor_layer.append_affine(tensor_w2, tensor_b2)
    tensor_layer.append_softmax()

    numpy_layers = []
    numpy_layers.append(common.layers.Affine(numpy_w1, numpy_b1))
    numpy_layers.append(common.layers.Relu())
    numpy_layers.append(common.layers.Affine(numpy_w2, numpy_b2))
    numpy_last_layer = common.layers.SoftmaxWithLoss()

    for i in range(loop_count):
        #forward
        t = time.time()
        tensor_forward = tensor_layer.forward(tensor_x).copy()
        print("tensor forward time : ", time.time() - t)
        t = time.time()
        numpy_X = numpy_x
        for layer in numpy_layers:
            numpy_X = layer.forward(numpy_X)
        numpy_loss = numpy_last_layer.forward(numpy_X, numpy_t)
        numpy_forward = numpy_last_layer.y
        print("numpy forward time : ", time.time() - t)


        #backward
        t = time.time()
        tensor_dout = tensor_layer.backward(tensor_t)
        tensor_loss = tensor_layer.layers[-1].loss
        print("tensor backward time : ", time.time() - t)
        
        t = time.time()
        numpy_dout = numpy_last_layer.backward(1)
        for i in range(len(numpy_layers)):
            numpy_dout = numpy_layers[-1 -i].backward(numpy_dout)
        print("numpy backward time : ", time.time() - t)

        #update
        t = time.time()
        tensor_layer.update(tensor.Tensor([0.1],[1]))
        print("tensor update time : ", time.time() - t)
        t = time.time()
        numpy_layers[0].W -= 0.1 * numpy_layers[0].dW
        numpy_layers[0].b -= 0.1 * numpy_layers[0].db
        numpy_layers[2].W -= 0.1 * numpy_layers[2].dW
        numpy_layers[2].b -= 0.1 * numpy_layers[2].db
        print("numpy update time : ", time.time() - t)

        print("loss")
        print("tensor : ",tensor_loss)
        print("numpy : ", numpy_loss)
        print("forward")
        print(compare(tensor_forward, numpy_forward, error))
        print("backward")
        print(compare(tensor_dout, numpy_dout, error))

        print("update")
        print("new w1")
        print(compare(tensor_layer.layers[0].W, numpy_layers[0].W, error))
        print("new b1")
        print(compare(tensor_layer.layers[0].b, numpy_layers[0].b, error))
        print("new w2")
        print(compare(tensor_layer.layers[2].W, numpy_layers[2].W, error))
        print("new b2")
        print(compare(tensor_layer.layers[2].b, numpy_layers[2].b, error))

def test_layer_with_batchNormalization(data_shape, layer1_shape, layer2_shape, loop_count, error = 0.001):
    tensor_x = tensor.create_randomly(data_shape)
    numpy_x = parseNumpy(tensor_x)
    
    tensor_t = tensor.create_zeros([data_shape[0], layer2_shape[-1]], int)
    for i in range(data_shape[0]):
        tensor_t.array[i * layer2_shape[-1] + rand.randint(0, layer2_shape[-1] - 1)] = 1

    numpy_t = parseNumpy(tensor_t)

    tensor_w1 = tensor.create_randomly(layer1_shape)
    tensor_b1 = tensor.create_randomly([layer1_shape[-1]])

    tensor_w2 = tensor.create_randomly(layer2_shape)
    tensor_b2 = tensor.create_randomly([layer2_shape[-1]])

    tensor_gamma1 = tensor.create_ones([layer1_shape[-1]])
    tensor_beta1 = tensor.create_zeros([layer1_shape[-1]])

    numpy_w1 = parseNumpy(tensor_w1)
    numpy_b1 = parseNumpy(tensor_b1)

    numpy_w2 = parseNumpy(tensor_w2)
    numpy_b2 = parseNumpy(tensor_b2)

    numpy_gamma1 = parseNumpy(tensor_gamma1)
    numpy_beta1 = parseNumpy(tensor_beta1)


    #layer
    import layer
    tensor_layer = layer.Layers()
    tensor_layer.append_affine(tensor_w1, tensor_b1)
    tensor_layer.append_batchnormalization(tensor_gamma1, tensor_beta1)
    tensor_layer.append_sigmoid()
    tensor_layer.append_affine(tensor_w2, tensor_b2)
    tensor_layer.append_softmax()

    numpy_layers = []
    numpy_layers.append(common.layers.Affine(numpy_w1, numpy_b1))
    numpy_layers.append(common.layers.BatchNormalization(numpy_gamma1, numpy_beta1))
    numpy_layers.append(common.layers.Sigmoid())
    numpy_layers.append(common.layers.Affine(numpy_w2, numpy_b2))
    numpy_last_layer = common.layers.SoftmaxWithLoss()

    tensor_layer.set_train_mode(True)
    
    for i in range(loop_count):
        #forward
        t = time.time()
        tensor_forward = tensor_layer.forward(tensor_x).copy()
        print("tensor forward time : ", time.time() - t)
        t = time.time()
        numpy_X = numpy_x
        for layer in numpy_layers:
            numpy_X = layer.forward(numpy_X)
        numpy_loss = numpy_last_layer.forward(numpy_X, numpy_t)
        numpy_forward = numpy_last_layer.y
        print("numpy forward time : ", time.time() - t)
        #print("pre_batch : ", tensor_layer.layers[0].out)
        #print("batch : ", tensor_layer.layers[1].out)

        #backward
        t = time.time()
        tensor_dout = tensor_layer.backward(tensor_t)
        tensor_loss = tensor_layer.layers[-1].loss
        print("tensor backward time : ", time.time() - t)
        
        t = time.time()
        numpy_dout = numpy_last_layer.backward(1)
        for i in range(len(numpy_layers)):
            numpy_dout = numpy_layers[-1 -i].backward(numpy_dout)
        print("numpy backward time : ", time.time() - t)

        #update
        t = time.time()
        tensor_layer.update(tensor.Tensor([0.1],[1]))
        print("tensor update time : ", time.time() - t)
        t = time.time()
        numpy_layers[0].W -= 0.1 * numpy_layers[0].dW
        numpy_layers[0].b -= 0.1 * numpy_layers[0].db
        numpy_layers[3].W -= 0.1 * numpy_layers[3].dW
        numpy_layers[3].b -= 0.1 * numpy_layers[3].db
        numpy_layers[1].gamma -= 0.1 * numpy_layers[1].dgamma
        numpy_layers[1].beta -= 0.1 * numpy_layers[1].dbeta
        print("numpy update time : ", time.time() - t)

        print("loss")
        print("tensor : ",tensor_loss)
        print("numpy : ", numpy_loss)
        print("forward")
        print(compare(tensor_forward, numpy_forward, error))
        print("backward")
        print(compare(tensor_dout, numpy_dout, error))

        print("update")
        print("new w1")
        print(compare(tensor_layer.layers[0].W, numpy_layers[0].W, error))
        print("new b1")
        print(compare(tensor_layer.layers[0].b, numpy_layers[0].b, error))
        print("new w2")
        print(compare(tensor_layer.layers[3].W, numpy_layers[3].W, error))
        print("new b2")
        print(compare(tensor_layer.layers[3].b, numpy_layers[3].b, error))
    return tensor_layer, numpy_layers
