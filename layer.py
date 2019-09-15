import tensor
import math

#교제의 layer.py보다 소스가 긴 이유는, 직접 만든 tensor 패키지 연산시, 연산 함수와 객체 생성 함수를 분리해서 명시적으로 객체 생성을 해야함.
#따라서 연산자 오버라이드를 사용할 수 없음.

class Layers:
    def __init__(self):
        self.layers = []
        self.enable_update_list = []
        self.enable_train_mode_list = []

    def append(self, layer):
        self.layers.append(layer)

    def append_update_list(self, layer):
        """layer에 업데이트가 필요한 W, b가 있는 경우"""
        self.layers.append(layer)
        self.enable_update_list.append(layer)

    def append_affine(self, w, b):
        affine = Affine(w,b)
        self.layers.append(affine)
        self.enable_update_list.append(affine)

    def append_relu(self):
        self.layers.append(Relu())

    def append_sigmoid(self):
        self.layers.append(Sigmoid())

    def append_softmax(self, exp_func = math.exp, log_func = math.log):
        self.layers.append(Softmax(exp_func, log_func))

    def append_batchnormalization(self, gamma, beta, momentum = 0.9, running_mean = None, running_var = None):
        batchnorm = BatchNormalization(gamma, beta, momentum, running_mean, running_var)
        self.layers.append(batchnorm)
        self.enable_update_list.append(batchnorm)
        self.enable_train_mode_list.append(batchnorm)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, table):
        self.layers[-1].init_table(table)
        dout = 1
        for i in range(len(self.layers)):
            dout = self.layers[-1-i].backward(dout)
        return dout

    def update(self, learning_rate):
        for layer in self.enable_update_list:
            layer.update(learning_rate)
        
    def set_train_mode(self, flg):
        for layer in self.enable_train_mode_list:
            layer.set_train_mode(flg)

    def _equal(left, right):
        if( left == right):
            return 1
        else:
            return 0
        
    def accuracy(self, table):

        """forward를 반드시 해야 하고, backward 이전에 사용해야 합니다."""
        out = self.layers[-1].out
        out_argmax = tensor.argmax(out, -1, tensor.create_sum(out, -1))
        table_argmax = tensor.argmax(table, -1, tensor.create_sum(table, -1))
        eq = tensor.function_elment_wise(out_argmax, table_argmax, Layers._equal, tensor.create_element_wise_product(out_argmax, table_argmax, int))
        reduce_sum = tensor.sum_axis(eq, 0, tensor.Tensor([1],[1]))
        return reduce_sum.array[0] / len(out_argmax.array)

#이 아래의 모든 layer는 공통적으로 똑같은 forward(self, x)와 backward(self, dout)가 있음.
#추가적으로 update가 필요한 layer는 update(self, learning_rate)가 있음
#추가적으로 마지막 레이어로 쓰이는 layer는 init_table(self, t)가 있음.

class Relu:
    def __init__(self):
        self.out = tensor.Tensor([0],[1,1])
        
    def relu_func(x):
        return x * (x > 0)

    def relu_dfunc(x,y):
        return x * (y > 0)

    def forward(self, x):
        # 1001개의 데이터를 100개씩 나누어 학습 할 경우, 마지막에는 1개의 데이터가 오기 때문.
        if(self.out.shape[-2] != x.shape[-2]):
            self.out = x.copy()
        # 지정된 함수로 텐서의 모든 원소들을 연산하는 함수.
        tensor.function(x, Relu.relu_func, self.out)
        return self.out

    def backward(self, dout):
        tensor.function_elment_wise(dout, self.out, Relu.relu_dfunc, self.out) # 기존 out값이 손실 됨.
        return self.out
    
class Sigmoid:
    def __init__(self):
        self.out = tensor.Tensor([0],[1,1])

    def sigmoid_func(x):
        return 1 / (1 + math.exp(-x))

    def forward(self, x):
        if(self.out.shape[-2] != x.shape[-2]):
            self.out = x.copy()
        tensor.function(x, Sigmoid.sigmoid_func, self.out)
        return self.out

    def backward(self, dout):
        out_array_len = len(self.out.array)
        for i in range(out_array_len):
            self.out.array[i] = dout.array[i] * (1 - self.out.array[i]) * self.out.array[i] #기존 forward의 out이 손실 됨.
        return self.out
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.dW = W.copy()
        self.W_t = tensor.create_transpose(W)
        self.b = b
        self.db = b.copy()
        self.out = tensor.Tensor([0],[1,1])
        self.dout = None
        self.x = None
        self.x_t = None

    def forward(self, x):
        if(self.out.shape[-2] != x.shape[-2]):
            self.x_t = tensor.create_transpose(x)
            self.out = tensor.create_matrix_product(x, self.W)
            self.dout = tensor.create_matrix_product(self.out, self.W_t)

        self.x = x # 읽기 전용!
        tensor.matmul(x, self.W, self.out)
        tensor.add(self.out, self.b, self.out) # 만약 self.b가 크면 tmp를 matmul과 add를 분리 하면 해결됨.
        return self.out

    def backward(self, dout):
        tensor.transpose(self.W, self.W_t)
        tensor.matmul(dout, self.W_t, self.dout) #기존 forward의 out이 손실 됨.
        tensor.transpose(self.x, self.x_t)
        tensor.matmul(self.x_t, dout, self.dW)
        tensor.sum_axis(dout, 0, self.db)
        return self.dout

    def update(self, learning_rate):
        def update_(x, delta_x):
            return x - learning_rate.array[0] * delta_x #learning_rate는 캡쳐됨.
        tensor.function_elment_wise(self.W, self.dW, update_, self.W)
        tensor.function_elment_wise(self.b, self.db, update_, self.b)

class Softmax:
    def __init__(self, exp_func = math.exp, log_func = math.log):
        self.exp_func = exp_func
        self.log_func = log_func
        self.loss = None
        self.y = None
        self.t = None
        self.out = tensor.Tensor([0],[1,1])
        self.tmp_sum = None
        self.batch_size = 0

    def overflow_process(x, out):
        #find max
        m = x.array[0]
        for e in x.array:
            if ( e > m ):
                m = e
        
        for i in range(len(x.array)):
            out.array[i] = x.array[i] - m

    def init_table(self, t):
        self.t = t
            
    def forward(self, x):
        if(self.out.shape[0] != x.shape[0]):
            self.out = x.copy()
            self.tmp_sum = tensor.create_sum(x, -1)
            self.batch_size = x.shape[0] #self.t.shape[0] # x.shape[0]

        #softmax
        SoftmaxWithLoss.overflow_process(x, self.out)
        tensor.function(self.out, self.exp_func, self.out)
        column_count = self.out.shape[-1]
        for r in range(self.batch_size):
            s = 0
            for c in range(column_count):
                s += self.out.array[r * column_count + c]
            for c in range(column_count):
                self.out.array[r * column_count + c] /= s

        return self.out

    def backward(self, dout):
        #cross_etropy_error
        column_count = self.t.shape[-1]
        not_one_hot = int(column_count == 1) #원핫이 아닌 경우 1
        self.loss = 0
        point = 0
        t_array_len = len(self.t.array)
        out_array_len = len(self.out.array)
        column_count = out_array_len // t_array_len#out는 당연 원 핫이고, table도 원 핫이면 길이가 같아 1이 되고 아니면, column 수가 됨.
        not_one_hot = 1 % column_count
        point = 0

        
        for i in range(t_array_len): #기존 forward의 out이 손실 됨.
            point = i * column_count
            
            for c in range(column_count):
                if(self.t.array[i] == 1 + c - not_one_hot):
                    self.loss -= self.log_func(self.out.array[point + c] + 1e-7)
                    self.out.array[point + c] = (self.out.array[point + c] - 1) / self.batch_size
                else:
                    self.out.array[point + c] = self.out.array[point + c] / self.batch_size

        self.loss /= self.batch_size
        return self.out

class SoftmaxWithLoss:
    def __init__(self, exp_func = math.exp, log_func = math.log):
        self.exp_func = exp_func
        self.log_func = log_func
        self.loss = None
        self.y = None
        self.t = None
        self.out = tensor.Tensor([0],[1])
        self.tmp_sum = None
        self.batch_size = 0

    def overflow_process(x, out):
        #find max
        m = x.array[0]
        for e in x.array:
            if ( e > m ):
                m = e
        
        for i in range(len(x.array)):
            out.array[i] = x.array[i] - m
            
    def forward(self, x, t):
        if(self.out.shape[0] != x.shape[0]):
            self.out = x.copy()
            self.tmp_sum = tensor.create_sum(x, -1)
            self.batch_size = t.shape[0] # x.shape[0]
            
        self.t = t

        #softmax
        SoftmaxWithLoss.overflow_process(x, self.out)
        tensor.function(self.out, self.exp_func, self.out)
        
        column_count = self.out.shape[-1]
        for r in range(self.batch_size):
            s = 0
            for c in range(column_count):
                s += self.out.array[r * column_count + c]
            for c in range(column_count):
                self.out.array[r * column_count + c] /= s

        #cross_etropy_error
        column_count = t.shape[-1]
        is_one_hot = int(column_count == 1) #원핫이 아닌 경우 1 변수명과 의미가 다름 주의...
        self.loss = 0
        point = 0

        for i in range(self.batch_size):
            for c in range(column_count):
                point = i * column_count + c
                if(t.array[point] + is_one_hot > 0):
                    self.loss -= self.log_func(self.out.array[point + is_one_hot * t.array[point]] + 1e-7)
                    break
        self.loss /= self.batch_size
        
        return self.loss

    def backward(self, dout=1):
        t_array_len = len(self.t.array)
        out_array_len = len(self.out.array)
        column_count = out_array_len // t_array_len#backward는 당연 원 핫이고, table도 원 핫이면 길이가 같아 1이 되고 아니면, column 수가 됨.
        not_one_hot = 1 % column_count
        point = 0
        
        for i in range(t_array_len): #기존 forward의 out이 손실 됨.
            point = i * column_count
            for c in range(column_count):
                if(self.t.array[i] == 1 + c - not_one_hot):
                    self.out.array[point + c] = (self.out.array[point + c] - 1) / self.batch_size
                else:
                    self.out.array[point + c] = self.out.array[point + c] / self.batch_size

        return self.out

class BatchNormalization:
    def __init__(self, gamma, beta, momentum = 0.9, running_mean = None, running_var = None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var
        
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.xn = None #직접 추가.
        self.std = None
        self.dgamma = None
        self.dbeta = None
        

        #학습 모드
        self.train_flg = None

        #기타 변수
        self.out = tensor.Tensor([0],[1])
        self.tmp_out_shape = None
        #self.tmp_sum_axis = None

        #forward의 self.running_mean is None 조건문과 이어 져야 함.
        if (self.running_mean != None):
            self.std = self.running_mean.copy()
            #self.tmp.sum_axis = self.running_mean.copy()
            self.dbeta = self.running_mean.copy()
            self.dgamma = self.running_mean.copy()
        
    def set_train_mode(self, flg):
        self.train_flg = flg

    def jegop(x):
        return x ** 2

    def sqrt(x):
        return math.sqrt(x + 10e-7)

    def sqrt_and_div(left, right):
        return left / math.sqrt(right + 10e-7)

    def forward(self, x):
        #최적화 기법
        def multiply_momentum_and_add(left, right):
            return left * self.momentum + (1 - self.momentum) * right#momentum 캡쳐
        
        if (self.out.shape[0] != x.shape[0]):
            self.xc = x.copy()
            self.xn = x.copy()
            self.out = x.copy()
            self.tmp_out_shape = x.copy()
            self.batch_size = x.shape[0]
            
        if self.running_mean is None:
            D = len(x.array) // x.shape[0]
            self.running_mean = tensor.create_zeros([D])
            self.running_var = tensor.create_zeros([D])
            self.std = self.running_mean.copy()
            #self.tmp_sum_axis = self.running_mean.copy()
            self.dbeta = self.running_mean.copy()
            self.dgamma = self.running_mean.copy()
            
        if self.train_flg:
            #tmp_sum_axis가 나중에 추가되서 값이 이상하게 나오면 바꾸자.
            tensor.mean_axis(x, 0, self.std)#std가 임시 객체로 활용. (mu에 해당)
            #self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu #(넘파이 버전)
            tensor.function_elment_wise(self.running_mean, self.std, multiply_momentum_and_add, self.running_mean)
            tensor.sub(x, self.std, self.xc)
            tensor.function(self.xc, BatchNormalization.jegop, self.xn) #xn도 계산에 필요한 임시 객체로 사용
            tensor.mean_axis(self.xn, 0, self.std) #std가 임시 객체로 활용(var에 해당)
            #self.running_var = self.momentum * self.running_var + (1-self.momentum) * var #넘파이 버전 알고리즘
            tensor.function_elment_wise(self.running_var, self.std, multiply_momentum_and_add, self.running_var)
            tensor.function(self.std, BatchNormalization.sqrt, self.std) # std가 가져야 할 값
            tensor.div(self.xc, self.std, self.xn)#xn이 가져야 할 값
        else:
            tensor.sub(x, self.running_mean, self.xc)
            tensor.function_elment_wise(self.xc, self.running_var, BatchNormalization.sqrt_and_div, self.xn)
        tensor.mul(self.gamma, self.xn, self.out)
        tensor.add(self.out, self.beta, self.out)
        return self.out

    def comput_dstd(dxn_xc, std):
        return - dxn_xc / (std * std)
    def comput_dvar(dstd, std):
        return 0.5 * dstd / std

    def backward(self, dout):
        #반복문 최소화하기 위한 함수
        def comput_new_dxc(dxc, xc_dvar):
            return dxc + (2.0 / self.batch_size) * xc_dvar
        def comput_dx(dxc, dmu):
            return dxc - dmu / self.batch_size
        
        tensor.mul(self.gamma, dout, self.out)# out를 dxn으로 사용(기존 forward out 손실)
        tensor.mul(self.out, self.xc, self.tmp_out_shape) #dstd를 구하기 전단계
        #반복문 최소화 기능.
        tensor.function_elment_wise(self.tmp_out_shape, self.std, BatchNormalization.comput_dstd, self.tmp_out_shape) #tmp_out_shape은 dstd(np.sum하기 전)로 사용
        tensor.function_elment_wise(self.tmp_out_shape, self.std, BatchNormalization.comput_dvar, self.tmp_out_shape) #tmp_out_shape는 dvar(np.sum하기 전)로 사용
        tensor.sum_axis(self.tmp_out_shape, 0, self.dbeta) #self.dbeta는 dvar로 사용(기존 dbeta값 손실)
        
        tensor.mul(self.xc, self.dbeta, self.xc) #xc를 xc와 dvar의 곱으로 사용(기존 xc값 손실) (dvar의 역할 끝)
        
        tensor.div(self.out, self.std, self.out)# out을 dxc로 사용 (dxn값 손실) (dxn 역할 끝)
        tensor.function_elment_wise(self.out, self.xc, comput_new_dxc, self.out)
        
        tensor.sum_axis(self.out, 0, self.dbeta) #dmu를 dbeta로 사용(기존 dvar값 손실)
        tensor.function_elment_wise(self.out, self.dbeta, comput_dx, self.out) #최종 backward값(dxn값 손실)
        
        
        tensor.sum_axis(dout, 0, self.dbeta)
        tensor.mul(self.xn, dout, self.tmp_out_shape) #tmp_out_shape는 dgamma를 구하기 위한 임시 객체로 재활용 (기존 dvar(np.sum하기 전)값 손실)
        tensor.sum_axis(self.tmp_out_shape, 0, self.dgamma)

        return self.out
    
    def update(self, learning_rate):
        #반복문 개선 버전 (최적화)
        def update_(x, delta_x):
            return x - learning_rate.array[0] * delta_x #learning_rate는 캡쳐됨.
        tensor.function_elment_wise(self.gamma, self.dgamma, update_, self.gamma)
        tensor.function_elment_wise(self.beta, self.dbeta, update_, self.beta)
        
