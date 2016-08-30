
# coding: utf-8

# Imports
# ======

# In[1]:

import pickle as pkl
import numpy as np
import theano
import theano.tensor as T
import lasagne 
from lasagne.layers import get_output_shape, InputLayer, DenseLayer, RecurrentLayer, NonlinearityLayer, ReshapeLayer, get_output, get_all_params, get_all_param_values, ElemwiseSumLayer
import ctc_cost
from time import time
from TIMIT_utils import index2char_TIMIT
from special_activations import clipped_relu
import sys
import tensorflow as tf

# Load data
# =========

# In[2]:

#f = open('TIMIT_data_prepared_for_CTC.pkl','rb')
f = open('dat_train.pkl','rb')
data = pkl.load(f)
inp = data['x']
inp1 = data['inputs']
msk = data['mask']
tgt = data['y_indices']
char = data['chars']


# Build the network
# =================

# In[3]:

input_size = len(inp1[0][0])
hidden_size = 300
num_output_classes = len(char)
learning_rate = 0.001
output_size = num_output_classes+1
batch_size = None
input_seq_length = None
gradient_clipping = 5


# Introduce the targets
# =====================

# In[5]:

# Define the Bi-RNN architecture
# ==============================

# In[6]:

l_in = tf.placeholder(tf.float32, shape=(batch_size, input_seq_length, input_size))

n_batch, n_time_steps, n_features = l_in.input_var.shape #Unnecessary in this version. Just collecting the info so that we can reshape the output back to the original shape
l_reshape1 = ReshapeLayer(l_in,(-1,input_size) )
h_1 = DenseLayer(l_reshape1, num_units=hidden_size, nonlinearity=clipped_relu)
l_reshape2 = ReshapeLayer(h_1,(n_batch,n_time_steps,hidden_size) )
l_rec_forward = RecurrentLayer(l_reshape2, num_units=hidden_size, grad_clipping=gradient_clipping, nonlinearity=clipped_relu)
l_rec_backward = RecurrentLayer(l_reshape2, num_units=hidden_size, grad_clipping=gradient_clipping, nonlinearity=clipped_relu, backwards=True)
l_rec_accumulation = ElemwiseSumLayer([l_rec_forward,l_rec_backward])
l_rec_reshaped = ReshapeLayer(l_rec_accumulation, (-1,hidden_size))
#l_h2 = DenseLayer(l_rec_reshaped, num_units=hidden_size, nonlinearity=clipped_relu)
l_out = DenseLayer(l_rec_reshaped, num_units=output_size, nonlinearity=lasagne.nonlinearities.linear)
l_out_reshaped = ReshapeLayer(l_out, (n_batch, n_time_steps, output_size))#Reshaping back
l_out_softmax = NonlinearityLayer(l_out, nonlinearity=lasagne.nonlinearities.softmax)
l_out_softmax_reshaped = ReshapeLayer(l_out_softmax, (n_batch, n_time_steps, output_size))


# Get the outputs
# ===============

# In[7]:

output_logits = get_output(l_out_reshaped)
output_softmax = get_output(l_out_softmax_reshaped)


# Collect all the parameters
# ==========================

# In[8]:

all_params = get_all_params(l_out,trainable=True)
# print all_params==[l_rec.W_in_to_hid, l_rec.b, l_rec.W_hid_to_hid, l_out.W, l_out.b]


# In[9]:

print 'Number of trainable parameters =', len(all_params)
print all_params==[l_rec_forward.W_in_to_hid, l_rec_forward.b, l_rec_forward.W_hid_to_hid, l_rec_backward.W_in_to_hid, l_rec_backward.b, l_rec_backward.W_hid_to_hid, l_out.W, l_out.b]


# Compute cost
# ============

# In[10]:

pseudo_cost = ctc_cost.pseudo_cost(y, output_logits)


# Compute gradients
# =================

# In[11]:

pseudo_cost_grad = T.grad(pseudo_cost.sum() / n_batch, all_params)


# Compute cost for evaluation
# ===========================

# In[12]:

true_cost = ctc_cost.cost(y, output_softmax)
cost = T.mean(true_cost)


# Calculate parameter updates
# ===========================

# In[14]:

shared_learning_rate = theano.shared(lasagne.utils.floatX(0.01))
updates = lasagne.updates.rmsprop(pseudo_cost_grad, all_params, learning_rate=learning_rate)


# Define the training op
# ======================

# In[15]:

theano.config.exception_verbosity='high'
train = theano.function([l_in.input_var,y], [output_logits, output_softmax, cost, pseudo_cost], updates=updates)


# Sanity check the input data
# ===========================

# In[16]:

inp0 = inp1[0]
inp00= np.asarray([inp0],dtype=theano.config.floatX)
tgt0 = np.asarray(tgt[0],dtype=np.int16)
tgt00 = np.asarray([tgt0])
print inp00.shape, tgt00.shape


# Run Training
# ============

# In[19]:

num_epochs = 100
#num_training_samples = len(inp1)
num_training_samples = 3000
for epoch in range(num_epochs):
    t = time()
    cost = 0
    failures = []

##### Step decay of learning rate
    if(epoch % 30 == 29 ):
	shared_learning_rate.set_value(shared_learning_rate.get_value() * 0.1 )
    
    for i in range(num_training_samples):
        curr_inp = inp1[i]
#         curr_msk = msk[i].astype(np.bool)
#         curr_inp = curr_inp[curr_msk]
        curr_inp = np.asarray([curr_inp],dtype=theano.config.floatX)
        curr_tgt = np.asarray(tgt[i],dtype=np.int16)
        curr_tgt = np.asarray([curr_tgt])
        try:
            _,_,c,_=train(curr_inp,curr_tgt)
            cost += c
        except IndexError:
            failures.append(i)
            print 'Current input seq: ', curr_inp
            print 'Current output seq: ', curr_tgt
            sys.exit(IndexError)
    f = open('result_3000samples_300param_new_arch','a')
    f.write('Epoch: '+ str(epoch) +'Cost: '+ str(float(cost/(num_training_samples-len(failures))))+ ', time taken ='+str( time()-t) +'\n')
    f.close()

    print 'Epoch: ', epoch, 'Cost: ', float(cost/(num_training_samples-len(failures))), ', time taken =', time()-t
#     print 'Exceptions: ', len(failures), 'Total examples: ', num_training_samples
    if epoch%10==0:        
        #Save the model
        np.savez('CTC_model_under_test_3000s_300p_new_arch.npz', *get_all_param_values(l_out_softmax_reshaped, trainable=True))
        for i in range(2):
            curr_inp = inp1[i]
            curr_inp = np.asarray([curr_inp],dtype=theano.config.floatX)
            curr_tgt = np.asarray(tgt[i],dtype=np.int16)
            curr_out = output_softmax.eval({l_in.input_var:curr_inp})
            print 'Predicted:', index2char_TIMIT(np.argmax(curr_out, axis=2)[0])
            print 'Target:', index2char_TIMIT(curr_tgt)


# In[20]:




# In[21]:



