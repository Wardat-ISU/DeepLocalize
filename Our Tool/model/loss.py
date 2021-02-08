import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)

class LossFuntion:    
    # loss
    def cross_entropy(self, labels,inputs):
        out_num = labels.shape[0]
        p = np.sum(labels.reshape(1,out_num)*inputs)
        loss = -np.log(p)
        return loss
    
    def binary_crossentropy(self,target, output, from_logits=False):
        if not from_logits:
            output = np.clip(output, 1e-7, 1 - 1e-7)
            output = np.log(output / (1 - output))
        return np.mean((target * -np.log(sigmoid(output)) + (1 - target) * -np.log(1 - sigmoid(output))))
    
    def categorical_crossentropy(self,target, output, from_logits=False):
        if not from_logits:
            output /= output.sum(axis=-1, keepdims=True)
        output = np.clip(output, 1e-7, 1 - 1e-7)
        return np.sum(target * -np.log(output), axis=-1, keepdims=False)
    
    def sum_squared_error(self,  targets, outputs, derivative=False ):
        if derivative:
            return outputs - targets 
        else:
            return 0.5 * np.mean(np.sum( np.power(outputs - targets,2), axis = 1 ))
    
    def mean_squared_error(self , y_true, y_pred):
        #y_true = np.cast(y_true, y_pred.dtype)
        return np.mean(np.square(y_pred - y_true), axis=-1)

    def mean_absolute_error(self, targets, outputs,  derivative=False ):
        return  np.mean(np.abs(outputs - targets), axis=-1)
