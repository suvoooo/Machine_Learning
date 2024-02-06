'''
accopanying code for {https://towardsdatascience.com/3-basic-concepts-in-neural-net-and-deep-learning-revisited-7f982bb7bb05}[why relu works]
'''

import numpy as np
import matplotlib.pyplot as plt


def f(x): return x**2;

x_in = np.linspace(-3, 3, 50)

x_out = f(x_in)
print ('check output array: ', x_out)


def node(a, x, b):
    '''
    define a relu function 
    potentially used as an activation for a node
    '''
    linear = a*x + b
    relu = np.maximum(0, linear)
    return relu
    

### imagine we have only two input to neural net    
### zero bias and a = 1, -1

y_out_1 = np.zeros((len(x_in), ))

# print (y_out_1) 

for i in range(len(x_in)):
    y_1 = node(1, x_in[i], 0) + node(-1, x_in[i], 0)
    y_out_1[i] = y_1
    
print (y_out_1)    


#### let's extend this for 4 inputs 
#### two same as before, the new two inputs have a=2 and -2 and bias -1

y_out_2 = np.zeros((len(x_in), ))

for i in range(len(x_in)):
    y_2 = node(1, x_in[i], 0) + node(-1, x_in[i], 0) + node(2, x_in[i], -2) + node(-2, x_in[i], -2)
    y_out_2[i] = y_2
    
print (y_out_2)
    
#### Extend a bit more fpr 6 inputs 
#### 4 same as before, the new two inputs have a=3 and -3 and bias -2

y_out_3 = np.zeros((len(x_in), ))

for i in range(len(x_in)):
    y_3 = node(1, x_in[i], 0) + node(-1, x_in[i], 0) + node(2, x_in[i], -2) + node(-2, x_in[i], -2) + node(3, x_in[i], -6) + node(-3, x_in[i], -6)
    y_out_3[i] = y_3


print (y_out_3)


fig = plt.figure(figsize=(8, 5))
plt.plot(x_in, x_out, ls='None', marker='*', color='green', alpha=0.6, label='f(x)')
plt.plot(x_in, y_out_1, ls='--', color='orange', alpha=0.5, label='Relu- 2 Inputs')
plt.plot(x_in, y_out_2, ls='-.', color='magenta', alpha=0.5, label='Relu- 4 Inputs')
plt.plot(x_in, y_out_3, ls=':', color='red', alpha=0.5, label='Relu- 6 Inputs')

plt.legend(fontsize=12)
plt.xlabel('Input')
plt.ylabel('Output')

plt.tight_layout()
plt.show()

### the job of the neural net is to update the weights ('a's) and biases ('b's) to replicate the input behaviour as much as possible. 
