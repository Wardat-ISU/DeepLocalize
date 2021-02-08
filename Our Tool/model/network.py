import numpy as np
import pickle 
import sys
from time import *
from model.loss import *
from model.layers import *
import numpy as np
import math
from numpy import zeros
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist 
import scipy.stats as st
import math
import scipy.stats as stats



def MeanResult (array):
    #print(np.mean(array))
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


def slope(x1, y1, x2, y2): 
    return (float)(y2-y1)/(x2-x1)

class Sequential:
    def __init__(self):

        self.layers = []
        self.optimizer =None
        self.loss=None
        self.metrics=None

   
    def add(self,layer):
        self.layers.append(layer)
    def layerNumber(self):
        self.lay_num = len(self.layers)
    def getLayerNumber(self):
        return self.lay_num 
    def compile(self, optimizer, loss, metrics=None):
        self.layerNumber()
        self.optimizer =optimizer
        self.loss=loss
        self.metrics=metrics
    def fit(self, training_data, training_label, batch_size, epoch):
        total_acc = 0
        start_time = time()
        checker = False
        iteration = 0
        Loss = [] 
        Forward =[]
        Backward =[]
        fiftyRound = 0
        ForwardCount = zeros([self.lay_num])
        BackwardCount = zeros([self.lay_num])
        
        ForwardMeanCount = zeros([self.lay_num])
        BackwardMeanCount = zeros([self.lay_num])
        for ls in range(self.lay_num):
            Forward.append([])
            Backward.append([])
            
        for e in range(epoch):
            for batch_index in range(0, training_data.shape[0], batch_size):
                # batch input
                if batch_index + batch_size < training_data.shape[0]:
                    data = training_data[batch_index:batch_index+batch_size]
                    label = training_label[batch_index:batch_index + batch_size]
                else:
                    data = training_data[batch_index:training_data.shape[0]]
                    label = training_label[batch_index:training_label.shape[0]]
                loss = 0
                acc = 0
                for b in range(len(data)):
                    x = data[b]
                    y = label[b]
                    # forward pass
                    for l in range(self.lay_num):
                        output = self.layers[l].forward(x,e,iteration)
                        result = output.flatten()
                        dist, p, par = get_best_distribution(result)
                        STD = dist.std(*par)
                        MAX = np.max(result)
                        MIN = np.min(result)
                        MEAN =np.mean(result)
                        if MEAN == 0 and self.layers[l].getName() !="Activation":
                            ForwardMeanCount[l] +=1
                        Forward[l].append(MEAN)
                        
                        if checker  and iteration%50 == 0 and MeanResult(Forward[l][50*fiftyRound:50*fiftyRound+50]):
                            end_time = time()
                            print("no chnage forward layer  \t epoch ={0} \t iteration ={1} \t layer name ={2} \t time ={3}".format(e,iteration,l,str(end_time - start_time)))
                            sys.exit(1)
                        #print(ForwardCount)
                        if  iteration > 50 and ForwardMeanCount[l] > (iteration/2) :
                            end_time = time()
                            print("no chnage forward layer  \t epoch ={0} \t iteration ={1} \t layer name ={2} \t time ={3}".format(e,iteration,l,str(end_time - start_time)))
                            sys.exit(1)
                        std  = str(STD)
                        P_value =str(p)

                        if math.isnan(MAX) or math.isnan(MIN) or math.isnan(MIN) or math.isnan(MEAN) :
                            end_time = time()
                            print("layer ={0}\t iteration ={1}\t epoch={2}\t time= {3}".format(l,iteration,e,(end_time-start_time)))
                            print("nan issue")
                            print("forward")
                            sys.exit(1)
                        if math.isinf(MAX) or math.isinf(MIN) or math.isinf(MIN) or math.isinf(MEAN) or  math.isinf(STD) or math.isinf(p):
                            end_time = time()
                            print("layer ={0}\t iteration ={1}\t epoch={2}\t time= {3}".format(l,iteration,e,(end_time-start_time)))
                            print("inf issue")
                            print("forward")
                            sys.exit(1)
                        x = output
                      
                    f = LossFuntion()
                    functionCall = getattr(f, self.loss)
                    lossResult = functionCall(y,output)
                    loss += lossResult
                    Loss.append(lossResult)
                    if checker:
                        Slope1 =slope(0,Loss[0],iteration,lossResult)
                        Slope2 =slope(iteration-1,Loss[iteration-1],iteration,lossResult)
                        distnace = lossResult - Loss[iteration-1]
                        if math.isinf(lossResult) or math.isinf(loss) or math.isinf(Slope1) or math.isinf(Slope2) or math.isinf(distnace):
                            end_time =time()
                            print("layer ={0}\t iteration ={1}\t time ={2}".format("Loss Function",iteration, (end_time-start_time)))
                            print("inf loss issue")
                            sys.exit(1)
                        if math.isnan(lossResult) or math.isnan(loss) or math.isnan(Slope1) or math.isnan(Slope2) or math.isnan(distnace):
                            end_time =time()
                            print("layer ={0}\t iteration ={1}\t time ={2}".format("Loss Function",iteration, (end_time-start_time)))
                            print("nan  loss issue")
                            sys.exit(1)
                    
                    
                    if self.loss == 'categorical_crossentropy':
                        acc += np.mean(np.equal(np.argmax(y, axis=-1), np.argmax(output, axis=-1)))
                        total_acc += np.mean(np.equal(np.argmax(y, axis=-1), np.argmax(output, axis=-1)))
                    else:
                        acc += np.mean(np.equal(y, np.round(output)))
                        total_acc += np.mean(np.equal(y, np.round(output)))
                        
                    if checker and (math.isnan(total_acc) or math.isnan(total_acc) or total_acc <= 0 ):
                        end_time =time()
                        print("calculate the ={0}\t iteration ={1}\t time ={2}".format("Accuracy Function",iteration, (end_time-start_time)))
                        #sys.exit(1)
                    # backward pass
                    dy = y
                    for l in range(self.lay_num-1, -1, -1):
                        dout = self.layers[l].backward(dy,e,iteration,self.optimizer)
                        result = dout.flatten()
                        dist, p, par = get_best_distribution(result)
                        STD = dist.std(*par)
                        MAX = np.max(result)
                        MIN = np.min(result)
                        MEAN =np.mean(result)
                        if MEAN == 0 and self.layers[l].getName() !="Activation":
                            BackwardMeanCount[l] +=1
                        Backward[l].append(MEAN)
                        
                        if checker  and iteration%50 == 0 and MeanResult(Backward[l][50*fiftyRound:50*fiftyRound+50]):
                            end_time = time()
                            print("no chnage forward layer  \t epoch ={0} \t iteration ={1} \t layer name ={2} \t time ={3}".format(e,iteration,l,str(end_time - start_time)))
                            sys.exit(1)
                        #print(BackwardCount)
                        if iteration > 50 and ForwardMeanCount[l] > (iteration/2):
                            end_time = time()
                            print("no chnage in backlayer  \t epoch ={0} \t iteration ={1} \t layer name ={2} \t time ={3}".format(e,iteration,l,str(end_time - start_time)))
                            sys.exit(1)
                        std  = str(STD)
                        P_value =str(p)
                        if math.isnan(MAX) or math.isnan(MIN) or math.isnan(MIN) or math.isnan(MEAN) :
                            end_time = time()
                            print("layer ={0}\t iteration ={1}\t epoch={2}\t time= {3}".format(l,iteration,e,(end_time-start_time)))
                            print("nan issue")
                            print("backward")
                            sys.exit(1)
                        if math.isinf(MAX) or math.isinf(MIN) or math.isinf(MIN) or math.isinf(MEAN) or  math.isinf(STD) or math.isinf(p):
                            end_time = time()
                            print("layer ={0}\t iteration ={1}\t epoch={2}\t time= {3}".format(l,iteration,e,(end_time-start_time)))
                            print("inf issue")
                            print("backward")
                            sys.exit(1)
                        dy = dout
                    
                    checker = True
                # increase # of iteration  
                   
                    iteration +=1
                    if iteration > 90 and iteration%50 == 0:
                        fiftyRound +=1
                        
                # result
                loss /= batch_size
                batch_acc = float(acc)/float(batch_size)
                training_acc = float(total_acc)/(float((batch_index+batch_size)) + ((training_data.shape[0])*e))
                #print("total",float(total_acc))
                #print("index",(float((batch_index+batch_size)*(e+1))))
                print('=== Epoch: {0:d}/{1:d} === Iter:{2:d} === Loss: {3:9} === BAcc: {4:.2f} === TAcc: {5:.2f} === '.format(e,epoch,batch_index+batch_size,str(loss),batch_acc,training_acc))
    
    def evaluate(self, data, label, test_size):
        total_acc = 0
        e = 0
        for i in range(test_size):
            x = data[i]
            y = label[i]
            for l in range(self.lay_num):
                output = self.layers[l].forward(x,e,i)
                x = output
            if self.loss == 'categorical_crossentropy':
                total_acc += np.mean(np.equal(np.argmax(y, axis=-1), np.argmax(output, axis=-1)))
            else:
                total_acc += np.mean(np.equal(y, np.round(output)))
        print('=== Test Size:{0:d} === Test Acc:{1:.2f} ==='.format(test_size, float(total_acc)/float(test_size)))


