import numpy as np
from dezero.core import Function
from dezero.core import as_variable, as_array
from dezero import utils
from dezero.core import Variable

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x,  = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x,  = self.inputs
        gx = gy * -sin(x)
        return gx
def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx
def tanh(x):
    return Tanh()(x)

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y

    def backward(self, gy):
        return transpose(gy)

def transpose(x):
    return Transpose()(x)

class Sum(Function):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(x)

class BroadcastTo(Function):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.shape)
        return gx
    
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):

    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy ,W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)

class MeanSquaredError(Function):
    
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).mean()
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / diff.size)
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None
    return y

class Linear(Function):
    
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        if len(self.inputs) ==3:
            x, W, b = self.inputs
            gb = sum_to(gy, b.shape)
        else:
            x, W = self.inputs
            gb = None
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)

def sigmoid_simple(x):
    x = as_variable(x)
    return 1 / (1 + exp(-x))

class Sigmoid(Function):
    
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)

class Relu(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x,  = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    return Relu()(x)
 
# dezero/functions.py
class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        gx = np.zeros(self.in_shape, dtype=gy.dtype)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return GetItem(self.slices)(ggx)


def get_item(x, slices):
    return GetItem(slices)(x)

class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        gx = gy / x
        return gx

def log(x):
    return Log()(x)

# functions.py
class Clip(Function):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        y = np.clip(x, self.min_val, self.max_val)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        mask = (x.data >= self.min_val) & (x.data <= self.max_val)
        gx = gy * mask
        return gx

def clip(x, min_val, max_val):
    return Clip(min_val, max_val)(x)


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = utils.softmax(x, self.axis)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = sum(gx, axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)

def cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y

class CrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        p = utils.softmax(x, axis=1)
        p = np.clip(p, 1e-15, 1.0)    # numpy clip，不是F.clip
        log_p = np.log(p)             # numpy log，不是F.log
        tlog_p = log_p[np.arange(N), t]
        y = -np.sum(tlog_p) / N       # np.sum，不是sum_to
        return y

    def backward(self, gy):
        x, t = self.inputs
        N = x.shape[0]
        p = softmax(x)
        t_onehot = np.zeros_like(p.data)
        t_onehot[np.arange(N), t.data] = 1
        gx = (p - t_onehot) * gy / N
        return gx

def cross_entropy(x, t):
    return CrossEntropy()(x, t)

def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))