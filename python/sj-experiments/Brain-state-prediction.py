# Use of basic embeddings for linear regression and predicting the brain states
# Author: Sharmistha

import sys
import numpy
import matplotlib.pylab as plt
sys.path.append("../../../hippocampus")
sys.path.append("../")
import load_data 
DEFAULT_PROC = 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'

def plot_mat(mat):
	fig = plt.figure()
	plt.imshow(mat,cmap='hot',aspect = 0.7)
	plt.colorbar(orientation='horizontal')
	fig.savefig("brain-data.pdf", bbox_inches='tight')

# try plotting by brain region

if __name__=='__main__':
	# loading data, test
	data_raw, labels, time = load_data.load_raw('A', 'verb','active')
	print 'data_raw', type(data_raw),"#:#", data_raw.shape,'\n'#<type 'numpy.ndarray'>  [words, re-attempts,sensors, time]
	print 'labels', type(labels),"#:#", len(labels),labels,'\n', # list
	print 'time', type(time), "#:#",time.shape,'\n' ##<type 'numpy.ndarray'> 

	# some settings, take as args later 

	subject = 'B'
	sen_type = 'active'
	experiment = 'krns2'
	DEFAULT_PROC = 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
	reps_to_use = 15  # default one to replicate Nicole's experiments
	num_instances = 1 # default one to replicate Nicole's experiments
	word='verb'

	data_raw, labels, time = load_data.load_raw(subject=subject,
                                                    word=word,
                                                    sen_type=sen_type,
                                                    experiment=experiment,
                                                    proc=DEFAULT_PROC)

	# averaging data 
	data, labels = load_data.avg_data(data_raw=data_raw,
                                      labels_raw=labels,
                                      experiment=experiment,
                                      num_instances=num_instances,
                                      reps_to_use=reps_to_use)

	print data.shape
	print data[0].shape
	print data[0].max(), 'maximum element of the array'
	print data[0].min(), 'minimum element of the array'
	plot_mat(data[0])
	print len(labels),'\n'
	print labels

	