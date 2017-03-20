'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict phoneme sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is basically a recreation of an experiment
on the TIMIT data set from chapter 7 of Alex Graves's book (Graves, Alex. Supervised Sequence 
Labelling with Recurrent Neural Networks, volume 385 of Studies in Computational Intelligence.
Springer, 2012.), minus the early stopping.

Author: Jon Rein
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from utils import data_lists_to_batches

INPUT_PATH = '../TRAIN/All/mfcc/' #directory of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = '../TRAIN/All/phone_y/' #directory of nPhonemes 1-D array .npy files


####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 300
batchSize = 128

####Network Parameters
nFeatures = 39 #12 MFCC coefficients + energy, and derivatives
nHidden = 512
nClasses = 30 #39 phonemes, plus the "blank" for CTC

####Load data
print('Loading data')
with open('TIMIT_data_prepared_for_CTC.pkl','rb') as f:
	data= pickle.load(f)
input_list = data['x']
target_list = data['y_indices']
charmap = data['chars']
print(charmap)
charmap.append('_')
batchedData, maxTimeSteps = data_lists_to_batches(input_list, target_list, batchSize)
totalN = len(input_list)

####Define graph
print('Defining graph')
graph = tf.Graph()
with tf.variable_scope('CTC'):
	with graph.as_default():

    ####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow
        
    ####Graph input
    		inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, batchSize, nFeatures))
    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #  Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
    		inputXrs = tf.reshape(inputX, [-1, nFeatures])
    #  Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    		inputList = tf.split(0, maxTimeSteps, inputXrs)
    		targetIxs = tf.placeholder(tf.int64)
    		targetVals = tf.placeholder(tf.int32)
    		targetShape = tf.placeholder(tf.int64)
    		targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    		seqLengths = tf.placeholder(tf.int32, shape=(batchSize))

    ####Weights & biases
    		weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    		biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
    		weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    		biasesOutH2 = tf.Variable(tf.zeros([nHidden]))
    		weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                                                     stddev=np.sqrt(2.0 / nHidden)))
    		biasesClasses = tf.Variable(tf.zeros([nClasses]))

    ####Network
    		forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    		backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    		fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32,
                                       scope='BDLSTM_H1')
    		fbH1rs = [tf.reshape(t, [batchSize, 2, nHidden]) for t in fbH1]
    		outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

    		logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

    ####Optimizing
    		logits3d = tf.pack(logits)
    		loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
    		optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

    ####Evaluating
    		logitsMaxTest = tf.slice(tf.argmax(logits3d,2), [0, 0], [seqLengths[0], 1])
    		predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
    		dense_pred = tf.sparse_tensor_to_dense(predictions)
    		errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / \
                tf.to_float(tf.size(targetY.values))

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    saver = tf.train.Saver()
    
    ckpt = tf.train.get_checkpoint_state('/users/TeamASR/models/')
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())

    
    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        batchErrors = np.zeros(len(batchedData))
        batchRandIxs = np.random.permutation(len(batchedData)) #randomize batch order
        for batch, batchOrigI in enumerate(batchRandIxs):
            batchInputs, batchTargetSparse, batchSeqLengths, batchTargetLists = batchedData[batchOrigI]
            batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
            [_, l, er, lmt, pred, logit] = session.run([optimizer, loss, errorRate, logitsMaxTest,dense_pred,logits3d], feed_dict=feedDict)
            print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
	    out1 = [charmap[i] for i in lmt ]
	    target1 = [ charmap[i]  for i in batchTargetLists[0] ]
	    o1 = ''.join(out1)
	    t1 = ''.join(target1)
	        #out1.remove('_')
            print('argmax Output: ' + o1)
	    print( 'Target: ' + t1)
            #print(pred[0,:])
	    print(pred[0].shape)
            #print(pred)
            #print(pred.shape)
	    #print(pred.indices)
	    #print(pred.shape)
	    #print(pred.values)
	    #array = pred.todense()
	    #array = tf.sparse_tensor_to_dense(pred[0],pred[1],pred[2])
	    	
            string = [ charmap[i] for i in pred[0] ]
            beam_out = ''.join(string)
            print('Beam Output: '+ beam_out)	    
	    #print(lmt)
	    #print(logits3d.shape)
	    #print(logits3d)
	    #print( pred.shape)
	    #print(pred[0].shape)
            #print(pred[0])
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batchErrors[batch] = er*len(batchSeqLengths)
        epochErrorRate = batchErrors.sum() / totalN
        print('Epoch', epoch+1, 'error rate:', epochErrorRate)
        if(epoch%20 ==19):
		save_path = saver.save(session, '/users/TeamASR/models/mfat@'+str(epoch+1))
		print('model saved in file: '+ save_path)

