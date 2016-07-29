import pickle as pkl
import numpy as np
import theano
import theano.tensor as T
import lasagne 
from lasagne.layers import InputLayer, DenseLayer, RecurrentLayer, NonlinearityLayer, ReshapeLayer, get_output, get_all_params, get_all_param_values, ElemwiseSumLayer
import ctc_cost
from time import time
from special_activations import clipped_relu
import sys

#Some parameters for the CLM
INPUT_SIZE = 29

#Hidden layer hyper-parameters
N_HIDDEN = 100
HIDDEN_NONLINEARITY = 'rectify'

#Gradient clipping
GRAD_CLIP = 100




def getTrainedRNN():
	''' Read from file and set the params (To Do: Refactor 
		so as to do this only once) '''
	input_size = 39
	hidden_size = 50
	num_output_classes = 29
	learning_rate = 0.001
	output_size = num_output_classes+1
	batch_size = None
	input_seq_length = None
	gradient_clipping = 5

	l_in = InputLayer(shape=(batch_size, input_seq_length, input_size))
	n_batch, n_time_steps, n_features = l_in.input_var.shape #Unnecessary in this version. Just collecting the info so that we can reshape the output back to the original shape
	# h_1 = DenseLayer(l_in, num_units=hidden_size, nonlinearity=clipped_relu)
	l_rec_forward = RecurrentLayer(l_in, num_units=hidden_size, grad_clipping=gradient_clipping, nonlinearity=clipped_relu)
	l_rec_backward = RecurrentLayer(l_in, num_units=hidden_size, grad_clipping=gradient_clipping, nonlinearity=clipped_relu, backwards=True)
	l_rec_accumulation = ElemwiseSumLayer([l_rec_forward,l_rec_backward])
	l_rec_reshaped = ReshapeLayer(l_rec_accumulation, (-1,hidden_size))
	l_h2 = DenseLayer(l_rec_reshaped, num_units=hidden_size, nonlinearity=clipped_relu)
	l_out = DenseLayer(l_h2, num_units=output_size, nonlinearity=lasagne.nonlinearities.linear)
	l_out_reshaped = ReshapeLayer(l_out, (n_batch, n_time_steps, output_size))#Reshaping back
	l_out_softmax = NonlinearityLayer(l_out, nonlinearity=lasagne.nonlinearities.softmax)
	l_out_softmax_reshaped = ReshapeLayer(l_out_softmax, (n_batch, n_time_steps, output_size))


	with np.load('first_working_CTC_model.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(l_out_softmax_reshaped, param_values, trainable = True)
	output = lasagne.layers.get_output( l_out_softmax_reshaped )
	return output

def getTrainedCLM():
	''' Read CLM from file '''
	
	l_in = lasagne.layers.InputLayer(shape = (None, None, INPUT_SIZE)) #One-hot represenntation of character indices
	l_mask = lasagne.layers.InputLayer(shape = (None, None))

	l_recurrent = lasagne.layers.RecurrentLayer(incoming = l_in, num_units=N_HIDDEN, mask_input = l_mask, learn_init=True, grad_clipping=GRAD_CLIP)
	Recurrent_output=lasagne.layers.get_output(l_recurrent)

	n_batch, n_time_steps, n_features = l_in.input_var.shape

	l_reshape = lasagne.layers.ReshapeLayer(l_recurrent, (-1, N_HIDDEN))
	Reshape_output = lasagne.layers.get_output(l_reshape)

	l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=INPUT_SIZE, nonlinearity = lasagne.nonlinearities.softmax)
	with np.load('CLM_model.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(l_dense, param_values,trainable = True)
	output = lasagne.layers.get_output( l_dense )
	return output

BiRNN = getTrainedRNN()
CLM = getTrainedCLM()