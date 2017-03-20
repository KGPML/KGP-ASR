from decode import *
import numpy as np
import os
import pickle
import numpy as np
import pickle
def wer(r, h):
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def main():
	TIMIT_pkl_file = os.path.join(os.getcwd(),'TIMIT_data_prepared_for_CTC.pkl')
	#TIMIT_pkl_file = os.path.join(os.getcwd(),'dat_test.pkl')

	with open(TIMIT_pkl_file,'rb') as f:
			data = pickle.load(f)
			list_of_alphabets = data['chars']
	tgt = data['y_indices']
	#sess = tf.Session()
	#ctcsess = tf.Session()
	#model,chars,vocab = loadCLM(sess)
	num_test = 10
	print "Model loaded"
	total_words = 0
	Clm_errors = 0
	argmax_errors = 0
	#perm = np.random.permutation(len(data['x']))
	perm = np.random.permutation(100)#1st 100 only
	test_vals = perm[:num_test]
	xtest = np.array([data['x'][i] for i in test_vals])
	xtest = np.transpose(xtest,[1,0,2])
	pred = runCTC(xtest)[0]
	print pred
	sess=tf.Session() #need new session 
	model,chars,vocab = loadCLM(sess)
	print(pred.shape)
	pred = np.transpose(pred,[1,0,2])
	for i in range(num_test):
		print(data['y_char'][i])
		print(decode(pred[i],sess,model,chars,vocab,list_of_alphabets))
		print('\n\n')
	sess.close()
	#indices = [list_of_alphabets.index('z'),list_of_alphabets.index('-'),list_of_alphabets.index(' ')]
	#for i in test_vals:
	#	
	#	input_data = data['x'][i];
	#	pred = ctcsess.run([logits3d],{inputX: [input_data],seqlength:np.array([776])})
	#	clm_decoded = decode(pred[0],clmsess,model,chars,vocab)
	#	#for tim in pred[0]:
		#	for de in indices:
		#		tim[de]=0
		#	tim[len(list_of_alphabets)]=0
	#	argmax_decoded = index2char_TIMIT(np.argmax(pred, axis = 2)[0])
	#	print "setence no. ", i
	#	print "clm_decoded : " , clm_decoded
	#	print "argmax_decoded : ",argmax_decoded 
	#	curr_tgt = np.asarray(tgt[i],dtype=np.int16)

	#	curr_tgt = index2char_TIMIT(curr_tgt)
	#	print "Target : ", curr_tgt
	#	total_words = total_words + len(curr_tgt.split())
	#	Clm_errors  = Clm_errors + wer(curr_tgt,clm_decoded)
	#	argmax_errors = argmax_errors + wer(curr_tgt,argmax_decoded)
	#	print "CLM word_error_rate :" ,float(Clm_errors)/total_words
	#	print "Argmax word_error_rate :",float(argmax_errors)/total_words
	#	print "total_words = ",total_words	
if __name__ =="__main__":
	main()
