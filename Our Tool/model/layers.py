import numpy as np 
import numpy as np 
import random
import sys
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist 
import scipy.stats as st
import math
import scipy.stats as stats
from time import *
from numpy import zeros
def MeanResult (array):
    if np.mean(array) == np.max(array):
        return True
    else:
        return False

def get_best_distribution(data):
    dist_names = ["norm"]#, "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        #print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    #print("Best fitting distribution: "+str(best_dist.name))
    #print("Best p value: "+ str(best_p))
    #print("Parameters for the best fit: "+ str(params[best_dist.name]))

    return best_dist, best_p, params[best_dist.name]

def entropy_prime(A2, Y):
    return (np.divide((A2-Y), A2*(1-A2)))

class Convolution2D:

    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):
        # weight size: (F, C, K, K)
        # bias size: (F) 
        self.F = num_filters
        self.K = kernel_size
        self.C = inputs_channel

        self.weights = np.zeros((self.F, self.C, self.K, self.K))
        self.bias = np.zeros((self.F, 1))
        for i in range(0,self.F):
            self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.C*self.K*self.K)), size=(self.C, self.K, self.K))

        self.p = padding
        self.s = stride
        self.lr = learning_rate
        self.name = name

    def zero_padding(self, inputs, size):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = 2 * size + w
        new_h = 2 * size + h
        out = np.zeros((new_w, new_h))
        out[size:w+size, size:h+size] = inputs
        return out

    def forward(self, inputs,e,iteration):
        # input size: (C, W, H)
        # output size: (N, F ,WW, HH)
        C = inputs.shape[0]
        W = inputs.shape[1]+2*self.p
        H = inputs.shape[2]+2*self.p
        self.inputs = np.zeros((C, W, H))
        for c in range(inputs.shape[0]):
            self.inputs[c,:,:] = self.zero_padding(inputs[c,:,:], self.p)
        WW = int((W - self.K)/self.s + 1)
        HH = int((H - self.K)/self.s + 1)
        feature_maps = np.zeros((self.F, WW, HH))
        for f in range(self.F):
            for w in range(WW):
                for h in range(HH):
                    feature_maps[f,w,h]=np.sum(self.inputs[:,w:w+self.K,h:h+self.K]*self.weights[f,:,:,:])+self.bias[f]

        return feature_maps

    def backward(self, dy,e,iteration,optimizer ='sgd'):

        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        F, W, H = dy.shape
        for f in range(F):
            for w in range(W):
                for h in range(H):
                    dw[f,:,:,:]+=dy[f,w,h]*self.inputs[:,w:w+self.K,h:h+self.K]
                    dx[:,w:w+self.K,h:h+self.K]+=dy[f,w,h]*self.weights[f,:,:,:]

        for f in range(F):
            db[f] = np.sum(dy[f, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def getName(self):
        return self.name

class Maxpooling2D:

    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.s = stride
        self.name = name

    def forward(self, inputs,e,iteration):
        self.inputs = inputs
        C, W, H = inputs.shape
        new_width = int((W - self.pool)/self.s + 1)
        new_height = int((H - self.pool)/self.s + 1)
        out = np.zeros((C, new_width, new_height))
        for c in range(C):
            for w in range(int(W/self.s)):
                for h in range(int(H/self.s)):
                    out[c, w, h] = np.max(self.inputs[c, w*self.s:w*self.s+self.pool, h*self.s:h*self.s+self.pool])
        return out

    def backward(self, dy,e,iteration,optimizer ='sgd'):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        
        for c in range(C):
            for w in range(0, W, self.pool):
                for h in range(0, H, self.pool):
                    st = np.argmax(self.inputs[c,w:w+self.pool,h:h+self.pool])
                    (idx, idy) = np.unravel_index(st, (self.pool, self.pool))
                    dx[c, w+idx, h+idy] = dy[c, int(w/self.pool), int(h/self.pool)]
        return dx

    def extract(self):
        return 
    
    def getName(self):
        return self.name
    
class Dense:
    def __init__(self, num_inputs, num_outputs, learning_rate, name, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        seed =7
        rnd = np.random.RandomState(seed)
        
        
        #self.weights = 0.01*np.random.rand(num_inputs, num_outputs)
        if kernel_initializer== 'normal':
            self.weights = rnd.normal(0.0, 0.05,(num_inputs, num_outputs))
            print("kernel_initializer = random_normal ")
            
        if kernel_initializer== 'uniform':
            self.weights = rnd.uniform(-0.05, 0.05, (num_inputs, num_outputs))
            print("kernel_initializer = uniform ")
            
        if kernel_initializer== 'glorot_uniform':
            self.weights = np.zeros((num_inputs, num_outputs))
            print("kernel_initializer = glorot_uniform ")
            nin = num_inputs; nout = num_outputs
            sd = np.sqrt(6.0 / (nin + nout))
            for i in range(nin):
                for j in range(nout):
                    self.weights[i,j] = np.float32(rnd.uniform(-sd, sd))
                    
            
        if kernel_initializer== 'glorot_normal':
            self.weights = np.zeros((num_inputs, num_outputs))
            print("kernel_initializer = glorot_normal")
            nin = num_inputs;  nout = num_outputs
            sd = np.sqrt(2.0 / (nin + nout))
            for i in range(nin):
                for j in range(nout):
                    self.weights[i,j]  = np.float32(rnd.uniform(0.0, sd))
                      
            
        if bias_initializer == 'zeros':
            self.bias = np.zeros((num_outputs, 1))
            print("bias_initializer = zero ")
        if bias_initializer == 'ones':
            self.bias = np.ones((num_outputs, 1))
            print("bias_initializer = ones ")
        
        self.lr = learning_rate
        self.name = name
        self.start_time = time()
        self.meanWeight = []
        self.meanDelt = []
        self.meanCount = 0
        self.deltCount = 0
        self.meanCountZero = 0
        self.meanDeltaZero = 0
        self.fiftyRound =0
    def forward(self, inputs,e,iteration):
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.bias.T

    def backward(self, dy,e,iteration,optimizer ='sgd'):
        if optimizer !='sgd' and optimizer != 'RMSprop' and optimizer != 'adam':
            print("please enter correct optimizer")
            sys.exit(1)
            
        if iteration > 90 and iteration%50 == 0:
            self.fiftyRound +=1
            
            
        self.v_w_1 = 0
        self.v_b_1 = 0
        self.vdW = 0
        self.vdb = 0
        self.sdW = 0
        self.vdW = 0
        self.sdb = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        #dy=np.array(dy)
        if dy.shape[0] == self.inputs.shape[0]:
            dy = dy.T
        #print(dy.shape)
        dw = dy.dot(self.inputs)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(dy.T, self.weights.T)
        
        # optimizer = SGD  
        if optimizer =='sgd':
            #print(optimizer)
            self.weights -= self.lr * dw.T
            self.bias -= self.lr * db
            
        
        # optimizer = RMSprop
        if optimizer == 'RMSprop':
            #print(optimizer)
            self.v_w_1 = self.beta1 * self.v_w_1 + (1 - self.beta1) * dw.T**2
            self.v_b_1 = self.beta1 * self.v_b_1 + (1 - self.beta1) * db**2
            self.weights -=  (self.lr/np.sqrt(self.v_w_1 + e+1)) * dw.T
            self.bias -=  (self.lr/np.sqrt(self.v_b_1 + e+1)) * db
            
        
        # optimizer = ADAM
        if optimizer == 'adam':
            #print(optimizer)
            self.vdW =  self.beta1 * self.vdW + (1 - self.beta1) * dw.T
            self.vdb =  self.beta1 * self.vdb + (1 - self.beta1) * db
            self.sdW = self.beta2 * self.sdW + (1 - self.beta2) * pow(dw.T, 2)
            self.sdb = self.beta2 * self.sdb + (1 - self.beta2) * pow(db, 2)
            
            sdw_corrected = self.sdW / (1-pow(self.beta2,e+1))
            sdb_corrected = self.sdb / (1-pow(self.beta2,e+1))
        
            vdw_corrected = self.vdW / (1-pow(self.beta1, e+1))
            vdb_corrected = self.vdb / (1-pow(self.beta1, e+1))
        
            self.weights = self.weights - (self.lr * vdw_corrected )/ (np.sqrt( sdw_corrected )+ self.epsilon)
            self.bias = self.bias - (self.lr * vdb_corrected )/ (np.sqrt( sdb_corrected )+ self.epsilon)
        
        W1 = self.weights.flatten()
        dist, p, par = get_best_distribution(W1)
        std = dist.std(*par)
        MAX = np.max(W1)
        MIN = np.min(W1)
        MEAN =np.mean(W1)
        if MEAN == 0:
            self.meanCountZero +=1
        self.meanWeight.append(MEAN)
        if iteration >0  and iteration%50 == 0 and MeanResult(self.meanWeight[50*self.fiftyRound:50*self.fiftyRound+50]):
            end_time = time()
            print("no chnage in Weight \t epoch ={0} \t iteration ={1} \t layer name ={2} \t time ={3}".format(e,iteration,self.name,str(end_time - self.start_time)))
            sys.exit(1)
        if iteration > 50 and self.meanCountZero > (iteration/2):
            end_time = time()
            print("no chnage in Weight \t epoch ={0} \t iteration ={1} \t layer name ={2} \t time ={3}".format(e,iteration,self.name,str(end_time - self.start_time)))
            sys.exit(1)
        STD  = str(std)
        P_value =str(p)
        if math.isnan(MAX) or math.isnan(MIN) or math.isnan(MIN) or math.isnan(MEAN):
            end_time = time()
            print("layer ={0}\t iteration ={1}\t epoch={2}\t time= {3}".format(self.name,iteration,e,(end_time-self.start_time)))
            print("nan issue")
            print("Weight")
            sys.exit(1)
        if math.isinf(MAX) or math.isinf(MIN) or math.isinf(MIN) or math.isinf(MEAN) or  math.isinf(std) or math.isinf(p):
            end_time = time()
            print("layer ={0}\t iteration ={1}\t epoch={2}\t time= {3}".format(self.name,iteration,e,(end_time-self.start_time)))
            print("inf issue")
            print("Weight")
            sys.exit(1)
        
        W1 = dw.flatten()
        dist, p, par = get_best_distribution(W1)
        std = dist.std(*par)
        MAX = np.max(W1)
        MIN = np.min(W1)
        MEAN =np.mean(W1)
        if MEAN == 0:
            self.meanDeltaZero +=1
        self.meanDelt.append(MEAN)
        if iteration >0  and iteration%50 == 0 and MeanResult(self.meanDelt[50*self.fiftyRound:50*self.fiftyRound+50]):
            end_time = time()
            print("no chnage in Weight \t epoch ={0} \t iteration ={1} \t layer name ={2} \t time ={3}".format(e,iteration,self.name,str(end_time - self.start_time)))
            sys.exit(1)
        if iteration > 50 and self.meanDeltaZero > (iteration/2):
            end_time = time()
            print("no chnage in delta Weight \t epoch ={0} \t iteration ={1} \t layer name ={2} \t time ={3}".format(e,iteration,self.name,str(end_time - self.start_time)))
            sys.exit(1)
        STD  = str(std)
        P_value =str(p)
        if math.isnan(MAX) or math.isnan(MIN) or math.isnan(MIN) or math.isnan(MEAN):
            end_time = time()
            print("layer ={0}\t iteration ={1}\t epoch={2}\t time= {3}".format(self.name,iteration,e,(end_time-self.start_time)))
            print("nan issue")
            print("delta Weight")
            sys.exit(1)
        if math.isinf(MAX) or math.isinf(MIN) or math.isinf(MIN) or math.isinf(MEAN) or  math.isinf(std) or math.isinf(p):
            end_time = time()
            print("layer ={0}\t iteration ={1}\t epoch={2}\t time= {3}".format(self.name,iteration,e,(end_time-self.start_time)))
            print("inf issue")
            print("delta Weight")
            sys.exit(1)
        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def getName(self):
        return self.name

class BatchNormalization:
    def __init__(self, gamma=1, beta=0, eps=0.001):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.name ="BatchNormalization"
        
    def forward(self, x,e,iteration):
        N, D = x.shape
        # compute per-dimension mean and std_deviation
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        # normalize and zero-center (explicit for caching purposes)
        x_mu = x - mean
        inv_var = 1.0 / np.sqrt(var + self.eps)
        x_hat = x_mu * inv_var
        # squash
        out = self.gamma * x_hat + self.beta
        self.inv_var = inv_var
        self.x_hat = x_hat
        self.x_mu = x_mu
        # cache variables for backward pass
        #cache = x_mu, inv_var, x_hat
        return out

    def backward(self, dout,e,iteration,optimizer ='sgd'):
        N, D = dout.shape
        #x_mu, inv_var, x_hat, gamma = cache
        # intermediate partial derivatives
        dxhat = dout * self.gamma
        dvar = np.sum((dxhat * self.x_mu * (-0.5) * (self.inv_var)**3), axis=0)
        dmu = (np.sum((dxhat * -self.inv_var), axis=0)) + (dvar * (-2.0 / N) * np.sum(self.x_mu, axis=0))
        dx1 = dxhat * self.inv_var
        dx2 = dvar * (2.0 / N) * self.x_mu
        dx3 = (1.0 / N) * dmu
        # final partial derivatives
        dx = dx1 + dx2 + dx3
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(self.x_hat*dout, axis=0)
        
        return dx #, dgamma, dbeta
    
    def getName(self):
        return self.name
class Flatten:
    def __init__(self):
        self.name ="Flatten"
        pass
    def forward(self, inputs,e,iteration):
        self.C, self.W, self.H = inputs.shape
        return inputs.reshape(1, self.C*self.W*self.H)
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        return dy.reshape(self.C, self.W, self.H)
    def extract(self):
        return
    
    def getName(self):
        return self.name

class Dropout():
    def __init__(self,prob=0.5):
        self.name ="Dropout"
        self.prob = prob
        self.params = []

    def forward(self,X,e,iteration):
        self.mask = np.random.binomial(1,self.prob,size=X.shape) / self.prob
        out = X * self.mask
        return out.reshape(X.shape)
    
    def backward(self,dout,e,iteration,optimizer ='sgd'):
        dX = dout * self.mask
        return dX
    
    def getName(self):
        return self.name
    
class ReLu:
    def __init__(self):
        self.name ="Activation"
        pass
    def forward(self, inputs,e,iteration):
        self.inputs = inputs
        ret = inputs.copy()
        ret[ret < 0] = 0
        return ret
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        dx = dy.copy()
        dx[self.inputs < 0] = 0
        return dx
    def extract(self):
        return
    
    def getName(self):
        return self.name

class LeakyReLU:
    def __init__(self,leakage):
        self.name ="Activation"
        self.leakage =leakage
    def forward(self, inputs,e,iteration):
        self.inputs = inputs
        return np.clip(inputs > 0, self.leakage, 1.0)
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        dx = dy.copy()
        dx[self.inputs < 0] *= self.leakage
        return dx
    def extract(self):
        return
    def getName(self):
        return self.name




class Softmax:
    def __init__(self):
        self.name ="Activation"
        pass
    def forward(self, inputs,e,iteration, axis=-1):
        exp = np.exp(inputs - np.max(inputs, axis, keepdims=True))
        self.out = exp/np.sum(exp)
        return self.out
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        return self.out.T - dy.reshape(dy.shape[0],1)
    def extract(self):
        return
    
    def getName(self):
        return self.name

class Tanh:
    def __init__(self):
        self.name ="Activation"
        pass
    def forward(self, inputs,e,iteration):
        self.out = np.tanh(inputs)
        return self.out
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        return self.out.T - dy.reshape(dy.shape[0],1)
    def extract(self):
        return
    
    def getName(self):
        return self.name
    
class Sigmoid:
    def __init__(self):
        self.name ="Activation"
        pass
    def forward(self, inputs,e,iteration):
        self.out = 1. / (1. + np.exp(-inputs))
        return self.out
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        return self.out.T - dy.reshape(dy.shape[0],1)
    def extract(self):
        return
    
    def getName(self):
        return self.name
    
class Linear:
    def __init__(self):
        self.name ="Activation"
        pass
    def forward(self, inputs,e,iteration):
        self.out = inputs
        return inputs
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        return  ( dy.reshape(dy.shape[0],1) - self.out.T)
    def extract(self):
        return
    def getName(self):
        return self.name

class ReLuLast:
    def __init__(self):
        self.name ="Activation"
        pass
    def forward(self, inputs,e,iteration):
        self.inputs = inputs
        ret = inputs.copy()
        ret[ret < 0] = 0
        self.out = ret
        return self.out
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        dx = self.out.copy()
        #dx = np.where(dy > 0, 1.0, 0.0)
        dx[self.out > 0] = 1
        return (dx - dx) * dx
    def extract(self):
        return
    
    def getName(self):
        return self.name
    
class SoftmaxINSID:
    def __init__(self):
        self.name ="Activation"
        pass
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs), dtype=np.float)
        self.out = exp/np.sum(exp)
        return self.out
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        signal = dy.copy()
        e_x = np.exp( signal - np.max(signal, axis=1, keepdims = True) )
        signal = e_x / np.sum( e_x, axis = 1, keepdims = True )
        return np.ones( signal.shape )
    def extract(self):
        return
    
    def getName(self):
        return self.name
    
class SigmoidINSID:
    def __init__(self):
        self.name ="Activation"
        pass
    def forward(self, inputs,e,iteration):
        self.inputs = inputs
        ret = inputs.copy()
        ret = 1. / (1. + np.exp(-ret))
        return ret
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        dx = dy.copy()
        return np.multiply(dx, 1 - dx)
    def extract(self):
        return
    
    def getName(self):
        return self.name

class TanhInside:
    def __init__(self):
        self.name ="Activation"
        pass
    def forward(self, inputs,e,iteration):
        self.out = np.tanh(inputs)
        return self.out
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        return 1.0 - np.square(np.tanh(dy))
    def extract(self):
        return
    def getName(self):
        return self.name

class Reshape:
    def __init__(self,X,Y):
        self.name ="Reshape"
        self.X =X
        self.Y=Y
        pass
    def forward(self, inputs,e,iteration):
        self.CC, self.WW, self.HH = inputs.shape
        return inputs.reshape(1, self.X*self.Y)
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        out = dy.reshape(self.CC, self.WW, self.HH)
        return out
    def extract(self):
        return
    
    def getName(self):
        return self.name
    
class LinearINSIDE:
    def __init__(self):
        self.name ="Activation"
        pass
    def forward(self, inputs,e,iteration):
        return inputs
    def backward(self, dy,e,iteration,optimizer ='sgd'):
        return np.ones( dy.shape )
    def extract(self):
        return
    def getName(self):
        return self.name