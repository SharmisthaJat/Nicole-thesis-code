# Predict the embeddings for verbs from various datasets, and compare the performance of various embeddings for the prediction

# Author: Sharmistha

import sys
sys.path.append("../../../hippocampus")
sys.path.append("../")
import load_data 
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
#DEFAULT_PROC = 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
DEFAULT_PROC = 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_band-1-150_notch-60-120_beatremoval-first_blinkremoval-first'
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import numpy as np
from sklearn.model_selection import cross_val_predict

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def eval_data_correlation_across_subjects():
	pass
def eval_data_correlation_across_active_and_passive_contexts():
	pass

def partition_data(data):
	if type(data) is list:
		return data[:int(len(data)*.8)],data[int(len(data)*.8):],[];
	else:
		return data[:int(data.shape[0]*.8),:],data[int(data.shape[0]*.8):,:],[];

def load_vectors_text(file_path):
    word2vec = {}
    with open(file_path) as lines:
        for line in lines:
            split = line.split(" ")
            word = split[0]
            vector_strings = split[1:]
            vector = [float(num) for num in vector_strings]
            word2vec[word] = np.array(vector)
    return word2vec

def load_vectors_bin(file_path):
	return gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

def load_dataset():
	subject = 'B'
	sen_type = 'active'
	experiment = 'krns2'
	#DEFAULT_PROC = 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
	reps_to_use = 15  # default one to replicate Nicole's experiments
	num_instances = 1 # default one to replicate Nicole's experiments
	word='verb'
	
	data_raw, labels, time = load_data.load_raw(subject=subject,
                                                    word=word,
                                                    sen_type=sen_type,
                                                    experiment=experiment,
                                                    proc=DEFAULT_PROC)
                                                  
	#data_raw, labels, time = load_data.load_raw_all_verbs()
	# averaging data 
	data, labels = load_data.avg_data_all(data_raw=data_raw,
                                      labels_raw=labels,
                                      num_instances=num_instances,
                                      reps_to_use=reps_to_use)
	return data,labels

def cal_cosine_similarity(a,b):
    return cosine_similarity(np.array(a).reshape(1, -1),np.array(b).reshape(1, -1))[0][0]

def make_rs_matrix(predicted_vec, candidate_label_vec,true_labels):
	print true_labels, 
	rs = [[0 for i in range(len(candidate_label_vec))] for j in range(predicted_vec.shape[0])] 
	for item_num,prediction in enumerate(predicted_vec):
		simi = [(candidate[0],cal_cosine_similarity(prediction,candidate[1]))for candidate in candidate_label_vec]
		# sort similarity in descending order
		simi_sorted = sorted(simi, key=itemgetter(1),reverse=True)
		# mark the rank of true item
		label_ranks = [a[0] for a in simi_sorted]
		print label_ranks, simi_sorted
		print len(rs),":",len(rs[0]),":",label_ranks.index(true_labels[item_num]),true_labels[item_num], item_num
		rs[item_num][label_ranks.index(true_labels[item_num])] = 1
	print rs
	return rs

if __name__=='__main__':
	# loading data, test
	data,labels =  load_dataset()
	print data.shape
	print labels

	data_flattened=data.reshape(data.shape[0],(data.shape[1]*data.shape[2]))

	xtrain, xtest, xdev = partition_data(data_flattened)
	#print 'loading w2v..'
	#emb = load_vectors_bin('/home/sjat/repositories/Nicole-thesis-code/python/sj-experiments/embeddings_files/GoogleNews-vectors-negative300.bin')

	emb = load_vectors_text('embeddings_files/glove.840B.300d.txt')
	label_emb = np.array([emb[l] for l in labels])
	possible_emb = [(l,emb[l]) for l in list(set(labels))]


	ytrain_emb, ytest_emb,ydev_emb = partition_data(label_emb)
	ytrain_label, ytest_label,ydev_label = partition_data(labels)

	# learn linear regressor for hopefully maximing the MRR for embedding of the target word 

	lregressor = LinearRegression()  # MRR:  0.703125 (w2v),  MRR:  0.791666666667(glove) // cv=8
	#lregressor = linear_model.Ridge (alpha = 2)  # 0.33 (w2v), 0.33(glove)
	lregressor = linear_model.Lasso(alpha = 0.1)  #  0.33 (w2v)
	# lregressor = linear_model.BayesianRidge() # (w2v), (glove), error
	#lregressor.fit(xtrain,ytrain_emb)
	predicted = cross_val_predict(lregressor, data_flattened, label_emb, cv=8)
	
	#ypredicted_emb = lregressor.predict(xtest)

	rs_matrix = make_rs_matrix(predicted,possible_emb,labels)
	#print 'rs_matrix', rs_matrix
	print 'MRR: ',mean_reciprocal_rank(rs_matrix)



