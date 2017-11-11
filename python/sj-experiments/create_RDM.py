# create a dis-similarity matrix in the RDM format
# Author: Sharmistha
import gensim
import sys
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append("../../../hippocampus")
sys.path.append("../")
import load_data 
import csv
import cPickle
import argparse
import nltk
from nltk.tag import StanfordPOSTagger
nltk.internals.config_java(options='-xmx2G')
#matplotlib.use('Agg') # TkAgg - only works when sshing from office machine
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
import string
import Mantel
from numpy.linalg import matrix_rank

EXP_INDS = {'krns2': range(64, 96),
           'PassAct2': range(32, 64),
           'PassAct3': range(32)}
LST_INDS = {'krns2-s': 64,'krns2-e': 96,
            'PassAct2-s': 32,'PassAct2-e': 64,
            'PassAct3-s': 0,'PassAct3-e': 32}

NUMAP = 32
DEFAULT_PROC = 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'

def plot_mat(mat,labels,plot_name,plot_inch_sz):
    fig,ax = plt.subplots()
    fig.set_size_inches(plot_inch_sz[0],plot_inch_sz[1])
    cax = ax.imshow(mat,cmap='hot')#,aspect = 0.7)
    
    cbar = fig.colorbar(cax, orientation='horizontal')
    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels,ha='right',rotation=45)
    ax.set_yticklabels(labels,rotation=45)
    fig.savefig(plot_name+".pdf", dpi=100)

def cal_cosine_similarity(a,b):
    return cosine_similarity(np.array(a).reshape(1, -1),np.array(b).reshape(1, -1))[0][0]

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

def get_corpus_words(kwargs={}):
    #kwargs['batch_size']
    subject = 'B'
    sen_type = 'active'
    experiment = 'krns2'
    DEFAULT_PROC = 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
    reps_to_use = 10  # default one to replicate Nicole's experiments
    num_instances = 2 # default one to replicate Nicole's experiments
    word='verb'

    data_raw, labels, time = load_data.load_raw(subject=subject,
                                                    word=word,
                                                    sen_type=sen_type,
                                                    experiment=experiment,
                                                    proc=DEFAULT_PROC)
    return labels

def sentence_average_emb(sentence,emb):
    sent = [a.strip() for a in sentence.split()]
    if len(sent) > 0:
        sent_emb=[]
        for w in sent:
            try:
                word = w.replace(".","")
                word = word.replace(",","")
                sent_emb.append(emb[word])
            except:
                continue
        if len(sent_emb)<1:
            return np.array([0]*300)
        return np.mean(np.array(sent_emb),axis=0)
    else:        
        return None

def sentence_average_emb_split(sentence,emb):
    sent = [a.strip() for a in sentence]
    if len(sent) > 0:
        sent_emb=[]
        for w in sent:
            try:
                word = w.replace(".","")
                word = word.replace(",","")
                sent_emb.append(emb[word])
            except:
                continue
        if len(sent_emb)>0:
            return np.mean(np.array(sent_emb),axis=0)
        else:
            print '------------ error --------- no embeddings found ', sentence, sent
            None
    else:
        return None

def get_corpus_sents(file_name):
    r = csv.reader(open(file_name),delimiter='\t')
    sent_list=[]
    for line in r:
        sent_list.extend(line)
    return sent_list

WAS = 'was'
BY = 'by'

#SEN_LENS = {6: 'A', 8: 'P', 4: 'AS', 5: 'PS'}  # Nicole's code
# updated code due to key error : 7
SEN_LENS = {6: 'A', 8: 'P', 4: 'AS', 5: 'PS',7:''}
KEY_WORDS = {'A': [1, 2, 4], 'P': [1, 3, 6], 'AS': [1, 2], 'PS': [1, 3],'':[]}



def syn_rdm(ap_list):
    ap_rdm = np.empty((NUMAP, NUMAP))
    for i, i_sen in enumerate(ap_list):
        if i >= NUMAP:
            break
        for j, j_sen in enumerate(ap_list):
            if j >= NUMAP:
                break
            if i_sen == j_sen:
                ap_rdm[i, j] = 0.0
            elif i_sen in j_sen or j_sen in i_sen:
                ap_rdm[i, j] = 0.5
            else:
                ap_rdm[i, j] = 1.0
    return ap_rdm


def sem_rdm(sen_list, ap_list):
    ap_rdm = np.empty((NUMAP, NUMAP))
    for i, i_sen in enumerate(sen_list):
        if i >= NUMAP:
            break
        key_words_i = [i_sen[i_word] for i_word in KEY_WORDS[SEN_LENS[len(i_sen)]]]
        for j, j_sen in enumerate(sen_list):
            if j >= NUMAP:
                break
            key_words_j = [j_sen[j_word] for j_word in KEY_WORDS[SEN_LENS[len(j_sen)]]]
            # print(key_words_i)
            # print(key_words_j)
            # print(list(reversed(key_words_j)))
            if i_sen == j_sen:
                ap_rdm[i, j] = 0.0
            elif (ap_list[i] != ap_list[j]) and (key_words_i == list(reversed(key_words_j))):
                ap_rdm[i, j] = 0.0
            elif (ap_list[i] in ap_list[j] or ap_list[j] in ap_list[i]) and (key_words_i[0] == key_words_j[0] and key_words_i[1] == key_words_j[1]):
                ap_rdm[i, j] = 0.5
            else:
                ap_rdm[i, j] = 1.0
            # print(ap_rdm[i, j])
    return ap_rdm

def create_semantic_rdm(sen_list,emb):
    sent_list_emb = []
    for sent in sen_list:
        sent_list_emb.append(sentence_average_emb_split(sent,emb))

    non_smilarity_matrix = np.zeros((len(sen_list),len(sen_list)))
    for i,s1 in enumerate(sent_list_emb):
        for j,s2 in enumerate(sent_list_emb):  
            if i==j:
                val = 0.0 
            elif s1 is None or s2 is None:
                val = 1.0            
            else:
                val = 1.0-cal_cosine_similarity(s1,s2)
            non_smilarity_matrix[i][j] =  val
    return non_smilarity_matrix

def create_noun_semantic_rdm(sen_list,emb):
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
    sent_list_emb = []
    for sent in sen_list:
        tags = st.tag(sent)
        words=[]
        for wordtag in tags:
            if wordtag[1].startswith('N'):
                words.append(wordtag[0])        
        sent_list_emb.append(sentence_average_emb_split(words,emb))
            
    non_smilarity_matrix = np.zeros((len(sen_list),len(sen_list)))
    for i,s1 in enumerate(sent_list_emb):
        for j,s2 in enumerate(sent_list_emb):  
            if i==j:
                val = 0.0  
            elif s1 is None or s2 is None:
                val = 1.0
            else:
                val = 1.0-cal_cosine_similarity(s1,s2)
            non_smilarity_matrix[i][j] =  val
    return non_smilarity_matrix

def create_firstnoun_semantic_rdm(sen_list,emb):
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
    sent_list_emb = []
    for sent in sen_list:
        tags = st.tag(sent)
        words=[]
        for wordtag in tags:
            if wordtag[1].startswith('NN'):
                words.append(wordtag[0])
                break        
        sent_list_emb.append(sentence_average_emb_split(words,emb))
            
    non_smilarity_matrix = np.zeros((len(sen_list),len(sen_list)))
    for i,s1 in enumerate(sent_list_emb):
        for j,s2 in enumerate(sent_list_emb):  
            if i==j:
                val = 0.0  
            elif s1 is None or s2 is None:
                val = 1.0
            else:
                val = 1.0-cal_cosine_similarity(s1,s2)
            non_smilarity_matrix[i][j] =  val
    return non_smilarity_matrix

def create_noun_agent_semantic_rdm(sen_list,emb):
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
    sent_list_emb = []
    for sent in sen_list:
        tags = st.tag(sent)
        words=[]
        for wordtag in tags:
            if wordtag[1].startswith('NN'):
                words.append(wordtag[0])
                if len(words)>1:
                    break
        sent_list_emb.append(sentence_average_emb_split(words,emb))
            
    non_smilarity_matrix = np.zeros((len(sen_list),len(sen_list)))
    for i,s1 in enumerate(sent_list_emb):
        for j,s2 in enumerate(sent_list_emb):  
            if i==j:
                val = 0.0  
            elif s1 is None or s2 is None:
                val = 1.0
            else:
                val = 1.0-cal_cosine_similarity(s1,s2)
            non_smilarity_matrix[i][j] =  val
    return non_smilarity_matrix

def create_verb_semantic_rdm(sen_list,emb):
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
    sent_list_emb = []
    for sent in sen_list:
        tags = st.tag(sent)
        words=[]
        for wordtag in tags:
            if wordtag[1].startswith('VB'):
                words.append(wordtag[0])        
        sent_list_emb.append(sentence_average_emb_split(words,emb))
            
    non_smilarity_matrix = np.zeros((len(sen_list),len(sen_list)))
    for i,s1 in enumerate(sent_list_emb):
        for j,s2 in enumerate(sent_list_emb):            
            if i==j:
                val = 0.0          
            elif s1 is None or s2 is None:
                val = 1.0
            else:
                val = 1.0-cal_cosine_similarity(s1,s2)
            non_smilarity_matrix[i][j] =  val
    return non_smilarity_matrix

def create_lstm_semantic_rdm(sent_vec):
    non_smilarity_matrix = np.zeros((sent_vec.shape[0],sent_vec.shape[0]))
    for i,s1 in enumerate(sent_vec):
        for j,s2 in enumerate(sent_vec):            
            if i==j:
                val = 0.0          
            else:
                val = 1.0-cal_cosine_similarity(s1,s2)
            non_smilarity_matrix[i][j] =  val
    return non_smilarity_matrix    

def get_sen_lists(file_name):
    ap_list = []
    sen_list = []
    with open(file_name) as f:
        for line in f:
            sen_list.append(string.split(line))
            if WAS in line:
                if BY in line:
                    ap_list.append('P')
                else:
                    ap_list.append('PS')
            else:
                if len(string.split(line)) == 4:
                    ap_list.append('AS')
                else:
                    ap_list.append('A')
    return ap_list, sen_list

# compute kendall tau-b for rdm1 and rdm2
def rank_correlate_rdms(rdm1, rdm2):
    diagonal_offset = -1 # exclude the main diagonal
    # shar, check
    NUM_WORDS = rdm1.shape[0]
    lower_tri_inds = np.tril_indices(NUM_WORDS,diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalues= kendalltau(rdm1[lower_tri_inds],rdm2[lower_tri_inds])        
    return rdm_kendall_tau, rdm_kendall_tau_pvalues

def calculate_correlation(vec_rdm,semantic_rdm,ap_rdm,rnng_rdm,lstm_rdm):
    print "vec_rdm: (shape,rank)", vec_rdm.shape, ":",matrix_rank(vec_rdm), ":(",emb_name,")"
    print "ap_rdm: (shape,rank)", ap_rdm.shape, ":",matrix_rank(ap_rdm)
    print "semantic_rdm: (shape,rank)", semantic_rdm.shape, ":",matrix_rank(semantic_rdm)
    print "rnng_rdm: (shape,rank)", rnng_rdm.shape, ":",matrix_rank(rnng_rdm)
    #print "vec_rdm: (shape,rank)", vec_rdm.shape, ":",matrix_rank(vec_rdm)

    ktau1, pval1 = rank_correlate_rdms(vec_rdm, ap_rdm)

    print('Syntactic Kendall tau is {} with pval {}'.format(ktau1, pval1))

    ktau2, pval2 = rank_correlate_rdms(vec_rdm, semantic_rdm)

    print('Semantic Kendall tau is {} with pval {}'.format(ktau2, pval2))

    ktau3, pval3 = rank_correlate_rdms(rnng_rdm, vec_rdm)

    print('RNNG Kendall tau is {} with pval {}'.format(ktau3, pval3))   

    ktau4, pval4 = rank_correlate_rdms(lstm_rdm, vec_rdm)

    print('LSTM Kendall tau is {} with pval {}'.format(ktau4, pval4))   

    print ('-------------------------------------------------')

    print ('{:.3f}({:0.1e}) & {:.3f}({:0.1e}) & {:.3f}({:0.1e}) & {:.3f}({:0.1e})'.format(ktau1,pval1,ktau2,pval2,ktau3,pval3,ktau4,pval4))
    #ktau, pval = rank_correlate_rdms(rnng_rdm, ap_rdm)

    #print('Syntactic vs RNNG Kendall tau is {} with pval {}'.format(ktau, pval))

    #ktau, pval = rank_correlate_rdms(rnng_rdm, semantic_rdm)

    #print('Semantic  vs RNNG Kendall tau is {} with pval {}'.format(ktau, pval))

    #ktau, pval = rank_correlate_rdms(ap_rdm, semantic_rdm)

    #print('Semantic  vs Syntactic Kendall tau is {} with pval {}'.format(ktau, pval))
    print('======')
    #----------------------  Mantel test 

    r1, p1, z1 = Mantel.test(vec_rdm, ap_rdm)

    print('Syntactic Pearson is {} with pval {} and zval {} from Mantel test'.format(r1, p1, z1))

    r2, p2, z2 = Mantel.test(vec_rdm, semantic_rdm)

    print('Semantic Pearson is {} with pval {} and zval {} from Mantel test'.format(r2, p2, z2))

    r3, p3, z3 = Mantel.test(vec_rdm, rnng_rdm)

    print('RNNG Pearson is {} with pval {} and zval {} from Mantel test'.format(r3, p3, z3))

    r4, p4, z4 = Mantel.test(vec_rdm, lstm_rdm)

    print('LSTM Pearson is {} with pval {} and zval {} from Mantel test'.format(r4, p4, z4))

    print ('-------------------------------------------------')

    print ('{:.3f}({:0.1e}) & {:.3f}({:0.1e}) & {:.3f}({:0.1e}) & {:.3f}({:0.1e})'.format(r1,p1,r2,p2,r3,p3,r4,p4))

    print ('-------------------------------------------------')
    print ('-------------------------------------------------')
    print ('-------------------------------------------------')

    # r, p, z = Mantel.test(rnng_rdm, ap_rdm)

    # print('Syntactic vs RNNG Pearson is {} with pval {} and zval {} from Mantel test'.format(r, p, z))

    # r, p, z = Mantel.test(rnng_rdm, semantic_rdm)

    # print('Semantic  vs RNNG Pearson is {} with pval {} and zval {} from Mantel test'.format(r, p, z))

    # r, p, z = Mantel.test(ap_rdm, semantic_rdm)

    # print('Semantic  vs Syntactic Pearson is {} with pval {} and zval {} from Mantel test'.format(r, p, z))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",help="data for RDM",choices=["krns2","PassAct2"])
    parser.add_argument("embedding_type",help="data for RDM",choices=["w2v","glove",'None'])
    parser.add_argument("remove_unk",help="yes/no choice for remove unk",choices=["with_unk","no_unk"])
    args = parser.parse_args()
    # load embeddings
    if args.embedding_type == 'w2v':
        emb_name='w2v';emb = load_vectors_bin('/home/sjat/repositories/Nicole-thesis-code/python/sj-experiments/embeddings_files/GoogleNews-vectors-negative300.bin')
    elif args.embedding_type == 'glove':
        emb_name='glove';emb = load_vectors_text('/home/sjat/repositories/Nicole-thesis-code/python/sj-experiments/embeddings_files/glove.840B.300d.txt')
    elif args.embedding_type == 'None':
        emb_name='lstm';
        pass

    if args.remove_unk == 'no_unk':
        UNK_INDS = {'krns2':[0, 2, 5, 7, 9, 11, 12, 13, 14, 15, 16, 18, 21, 23, 25, 27, 28, 29, 30, 31],
            'PassAct2': [2, 5, 8, 15, 18, 21, 24, 31]
            }
        NON_UNK_INDS = {'krns2':[1, 3, 4, 6, 8, 10, 17, 19, 20, 22, 24, 26],
                    'PassAct2':[0, 1, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30]}
        if args.dataset == 'krns2':
            NUMAP = 12
        else:
            NUMAP = 24
    else:
        UNK_INDS = {'krns2':[],
            'PassAct2': []
            }
        NON_UNK_INDS = {'krns2':[],
            'PassAct2': []
            }

    # if args.dataset == 'word':
    #     # get words from the brain-data-corpus
    #     # words = get_corpus_words()
    #     words = [u'kicked', u'touched',u'found', u'inspected'] 
    #     words =list(set(words))
    #     print type(words),words,len(words)
        
    #     # build non-smilarity matrix
    #     non_smilarity_matrix = np.zeros((len(words),len(words)))
        
    #     for i,w1 in enumerate(words):
    #         for j,w2 in enumerate(words):
    #             non_smilarity_matrix[i][j] =  1-cal_cosine_similarity(emb[w1],emb[w2])

    #     file_name = './output/'+args.dataset+'-'+args.remove_unk+'--verb-disimilarity-'+emb_name
    #     plot_inch_sz = (5,5)
    #     plot_mat(non_smilarity_matrix,words,file_name,plot_inch_sz)
    #     cPickle.dump([words,[emb[w] for w in words],non_smilarity_matrix],open(file_name+".pkl",'w'))

    # for removing anything with UNK, remove items from EXP_INDS[args.dataset]


    
    # read csv and calculate sentence similarity with w2v averaging to get sentence context
    #sent_list = get_corpus_sents('./data/active-passive-sents.tsv')
    
    ap_list, sen_list = get_sen_lists('./data/sentence_stimuli_tokenized_tagged_with_unk_final.txt')

    # unk_list=[]
    # for i,item in enumerate(sen_list[LST_INDS[args.dataset+'-s']:LST_INDS[args.dataset+'-e']]):
    #     for itm in item:
    #         if 'UNK-LC' in itm:
    #             unk_list.append(i)
    #             break
    # unk_list = list(set(range(32))-set(unk_list))
    # print unk_list, 'non-unklist'
    # sys.exit(0)

    ap_list = [ap_list[LST_INDS[args.dataset+'-s']:LST_INDS[args.dataset+'-e']][i] for i in NON_UNK_INDS[args.dataset]]
    sen_list = [sen_list[LST_INDS[args.dataset+'-s']:LST_INDS[args.dataset+'-e']][i] for i in NON_UNK_INDS[args.dataset]]

    VECTORS = './data/sentence_stimuli_tokenized_tagged_pred_trees_no_preterms_vectors.txt'
    vectors = np.loadtxt(VECTORS)
    vectors = vectors[EXP_INDS[args.dataset], :][NON_UNK_INDS[args.dataset],:]
    rnng_rdm = squareform(pdist(vectors))

    print len(ap_list),len(sen_list), 'ap and sen list length'
    ap_rdm = syn_rdm(ap_list)
    semantic_rdm = sem_rdm(sen_list, ap_list)        
    sent_list = [" ".join(a) for a in sen_list]
    #-------------------------
    # plot rnng rdm
    file_name = './output/'+args.dataset+'-'+args.remove_unk+'--rnng-disimilarity-'
    plot_inch_sz=(26, 26)
    plot_mat(rnng_rdm,sent_list,file_name,plot_inch_sz)
    cPickle.dump([sent_list,rnng_rdm],open(file_name+".pkl",'w'))
    file_name = './output/'+args.dataset+'-'+args.remove_unk+'--semantic-disimilarity-'
    plot_inch_sz=(26, 26)
    plot_mat(semantic_rdm,sent_list,file_name,plot_inch_sz)
    cPickle.dump([sent_list,semantic_rdm],open(file_name+".pkl",'w'))
    file_name = './output/'+args.dataset+'-'+args.remove_unk+'--syntatic-disimilarity-'
    plot_inch_sz=(26, 26)
    plot_mat(ap_rdm,sent_list,file_name,plot_inch_sz)
    cPickle.dump([sent_list,ap_rdm],open(file_name+".pkl",'w'))
    #-----------------------------------------------------------
    print('with LSTM emb ------ --- --- ')
    sen_emb = np.loadtxt('./data/lstm_lm/test_sents_vectors.txt')[EXP_INDS[args.dataset], :][NON_UNK_INDS[args.dataset],:]
    lstm_rdm = create_lstm_semantic_rdm(sen_emb)
    vec_rdm = lstm_rdm
    calculate_correlation(vec_rdm,semantic_rdm,ap_rdm,rnng_rdm,lstm_rdm)

    file_name = './output/'+args.dataset+'-'+args.remove_unk+'--lstm-disimilarity-'+emb_name
    plot_inch_sz=(26, 26)
    plot_mat(vec_rdm,sent_list,file_name,plot_inch_sz)
    cPickle.dump([sent_list,vec_rdm],open(file_name+".pkl",'w'))
    #-----------------------------------------------------------
    print('with RNNG emb ------ --- --- ')
    calculate_correlation(rnng_rdm,semantic_rdm,ap_rdm,rnng_rdm,lstm_rdm)
    #-----------------------------------------------------------
    print('with full sentence average emb ------ --- --- ')
    non_smilarity_matrix = create_semantic_rdm(sen_list,emb)        
    vec_rdm = non_smilarity_matrix
    calculate_correlation(vec_rdm,semantic_rdm,ap_rdm,rnng_rdm,lstm_rdm)

    file_name = './output/'+args.dataset+'-'+args.remove_unk+'--sent-disimilarity-'+emb_name
    plot_inch_sz=(26, 26)
    plot_mat(vec_rdm,sent_list,file_name,plot_inch_sz)
    cPickle.dump([sent_list,non_smilarity_matrix],open(file_name+".pkl",'w'))
    #-----------------------------------------------------------
    print('with noun ------ --- --- ')
    noun_disimilarity_matrix = create_noun_semantic_rdm(sen_list,emb)
    vec_rdm = noun_disimilarity_matrix
    calculate_correlation(vec_rdm,semantic_rdm,ap_rdm,rnng_rdm,lstm_rdm)

    file_name = './output/'+args.dataset+'-'+args.remove_unk+'--noun-disimilarity-'+emb_name
    plot_inch_sz=(26, 26)
    plot_mat(vec_rdm,sent_list,file_name,plot_inch_sz)
    cPickle.dump([sent_list,vec_rdm],open(file_name+".pkl",'w'))
    #-----------------------------------------------------------
    print('with verb ------ --- --- ')
    verb_disimilarity_matrix = create_verb_semantic_rdm(sen_list,emb)
    vec_rdm = verb_disimilarity_matrix
    calculate_correlation(vec_rdm,semantic_rdm,ap_rdm,rnng_rdm,lstm_rdm)

    file_name = './output/'+args.dataset+'-'+args.remove_unk+'--verb-disimilarity-'+emb_name
    plot_inch_sz=(26, 26)
    plot_mat(vec_rdm,sent_list,file_name,plot_inch_sz)
    cPickle.dump([sent_list,vec_rdm],open(file_name+".pkl",'w'))
    #-----------------------------------------------------------
    print('with first noun only ------ --- --- ')
    first_noun_vec_disimilarity_matrix = create_firstnoun_semantic_rdm(sen_list,emb)
    vec_rdm = first_noun_vec_disimilarity_matrix
    calculate_correlation(vec_rdm,semantic_rdm,ap_rdm,rnng_rdm,lstm_rdm)

    file_name = './output/'+args.dataset+'-'+args.remove_unk+'--first-noun-disimilarity-'+emb_name
    plot_inch_sz=(26, 26)
    plot_mat(vec_rdm,sent_list,file_name,plot_inch_sz)
    cPickle.dump([sent_list,vec_rdm],open(file_name+".pkl",'w'))
    
     #-----------------------------------------------------------
    ap_list, sen_list = get_sen_lists('./data/sentence_agent.txt')
    ap_list = [ap_list[LST_INDS[args.dataset+'-s']:LST_INDS[args.dataset+'-e']][i] for i in NON_UNK_INDS[args.dataset]]
    sen_list = [sen_list[LST_INDS[args.dataset+'-s']:LST_INDS[args.dataset+'-e']][i] for i in NON_UNK_INDS[args.dataset]]
    sent_list = [" ".join(a) for a in sen_list]
    print('with agent emb ------ --- --- ')
    non_smilarity_matrix = create_semantic_rdm(sen_list,emb)        
    vec_rdm = non_smilarity_matrix
    calculate_correlation(vec_rdm,semantic_rdm,ap_rdm,rnng_rdm,lstm_rdm)

    file_name = './output/'+args.dataset+'-'+args.remove_unk+'--agent-disimilarity-'+emb_name
    plot_inch_sz=(26, 26)
    plot_mat(vec_rdm,sent_list,file_name,plot_inch_sz)
    cPickle.dump([sent_list,non_smilarity_matrix],open(file_name+".pkl",'w'))


