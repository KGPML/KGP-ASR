from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import argparse
import time
import os
from six.moves import cPickle

from utils_clm import TextLoader
from model import Model
import re
from six import text_type
import pickle
from utils import *
def loadCLM(sess):
    with open('/users/TeamASR/char-rnn-tensorflow/save/config.pkl', 'rb') as f:
        saved_args = cPickle.load(f)
    with open('/users/TeamASR/char-rnn-tensorflow/save/chars_vocab.pkl', 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state('/users/TeamASR/char-rnn-tensorflow/save')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    return model,chars,vocab

def pred(model,sess,chars,vocab,string,state):
	string=re.sub('_',' ',string)
	nparr,fin_state =  model.predict(sess, chars, vocab,string,state)
	nparr = nparr[0]
	dic = {}
	for char,ind in vocab.items():
		dic[char] = nparr[ind]
	return dic,fin_state



####Ugly. Should clean up later
def runCTC(batch):
    INPUT_PATH = '../TRAIN/All/mfcc/' #directory of MFCC nFeatures x nFrames 2-D array .npy files
    TARGET_PATH = '../TRAIN/All/phone_y/' #directory of nPhonemes 1-D array .npy files


    ####Learning Parameters
    learningRate = 0.001
    momentum = 0.9
    nEpochs = 300
    batchSize = batch.shape[1]

    ####Network Parameters
    nFeatures = 39 #12 MFCC coefficients + energy, and derivatives
    nHidden = 256
    nClasses = 30 #39 phonemes, plus the "blank" for CTC

    ####Load data
    print('Loading data')
    with open('TIMIT_data_prepared_for_CTC.pkl','rb') as f:
        data= pickle.load(f)
    input_list = batch
    charmap = data['chars']
    print(charmap)
    charmap.append('_')
    #batchedData, maxTimeSteps = data_lists_to_batches(input_list, target_list, batchSize)
    maxTimeSteps = 776
    totalN = len(input_list)

    ####Define graph
    print('Defining graph')
    graph = tf.Graph()
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
        errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / \
                    tf.to_float(tf.size(targetY.values))

    ####Run session
    with tf.Session(graph=graph) as session:
        print('Initializing')
        saver = tf.train.Saver()
        
        ckpt = tf.train.get_checkpoint_state('/users/TeamASR/models')
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
        feedDict = {inputX: batch, seqLengths: (np.ones([batchSize])*776)}
        logit = session.run([logits3d], feed_dict=feedDict)
    return logit

def decode( input_data,sess,model,chars,vocab,list_of_alphabets ):
    p_b = {}
    p_nb = {}
    p_tot = {}
    alpha = 0
    beta = 1.5
    k = 10
    p_ctc=[]
    for i in range(len(input_data)):
        dic = {}
        for j in range(len(list_of_alphabets)):
            dic[list_of_alphabets[j]] = input_data[i][j]
        dic["_"]=input_data[i][len(list_of_alphabets)] # _ is the last output
        p_ctc.append(dic)
    #"_" => Empty String from the psuedocode in the lexfree paper
    p_b["_"] = 1 
    p_nb["_"] = 0
    p_tot["_"] = 1
    z_prev = ["_"]
    state_dict = {}
    print("Number of frames:"+str(len(input_data)))
    for t in range(len(input_data)):
        #if(t % 200 == 0):
	print(str(t)+"/"+str(len(input_data))+" done")
        z_next = []
        p_b_next={}
        p_nb_next = {}
        for string in z_prev:
            print(string)
            p_clm,state = pred(model,sess,chars,vocab,string,state_dict)
            state_dict[string] = state
            p_b_next[string] = p_ctc[t]["_"] * p_tot[string]
            p_nb_next[string] = p_ctc[t][string[-1]] * p_nb[string]
            z_next.append(string)
            for char in list_of_alphabets:
                new_string = string + char
                if(char != string[-1]):
                    p_nb_next[new_string] = p_ctc[t][char]*(p_clm[char]**alpha)*p_tot[string]
                else:
                    p_nb_next[new_string] = p_ctc[t][char]*(p_clm[char]**alpha)*p_b[string]
                if( new_string not in p_b_next ):
                    p_b_next[new_string] = 0
                z_next.append(new_string)
        p_tot = {}
        plen = {}
        for string in z_next:
            p_tot[string] = p_b_next[string] + p_nb_next[string]
            plen[string] = p_tot[string]*(len(string)**beta)
        p_b = p_b_next
        p_nb = p_nb_next
        z_prev = sorted(plen, key=plen.get)[-k:]#get max k keys
    d=[]
    for i in z_prev:
        d.append((p_b[i]+p_nb[i]))
    ind=d.index(max(d))
    ans=z_prev[ind]
    return ans

if __name__ == '__main__':
	with tf.Session() as sess:
		model,chars,vocab = loadCLM(sess)
		prediction = pred(model,sess,chars,vocab,"she washed yo")
		print(prediction)
