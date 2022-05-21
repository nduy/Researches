#!/usr/bin/python
# -*- coding: utf-8 -*-
# encoding=utf8
"""
Created on Fri Nov 17 14:22:42 2017

@author: duy
"""
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import nltk
from nltk.stem import WordNetLemmatizer
from itertools import permutations
from sklearn import metrics
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# create sample documents
n_topics = 13


# list for tokenized documents in loop
texts = []


# Training
class Predict():
    def __init__(self, dictionary=None, model=None):
        # dictionary_path = "models/dictionary.dict"
        # lda_model_path = "models/lda_model_50_topics.lda"
        # self.dictionary = corpora.Dictionary.load(dictionary_path)
        # self.lda = LdaModel.load(lda_model_path)
        self.dictionary = dictionary
        self.lda = model
        
    def load_stopwords(self):
        stopwords = en_stop

        return stopwords

    def extract_lemmatized_nouns(self, new_review):
        stopwords = self.load_stopwords()
        words = []

        sentences = nltk.sent_tokenize(new_review.lower())
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            text = [word for word in tokens if word not in stopwords]
            tagged_text = nltk.pos_tag(text)

            for word, tag in tagged_text:
                words.append({"word": word, "pos": tag})

        lem = WordNetLemmatizer()
        nouns = []
        for word in words:
            if word["pos"] in ["NN", "NNS"]:
                nouns.append(lem.lemmatize(word["word"]))

        return nouns

    def run(self, new_review):
        nouns = self.extract_lemmatized_nouns(new_review)
        new_review_bow = self.dictionary.doc2bow(nouns)
        new_review_lda = self.lda[new_review_bow]

        return new_review_lda


def find_best_row_permutation(mat=None, rows=[], cols=[]):
    """
    Find the best order or rows, such that sum of dianog is maximum. The idea
    is first start from the highest overlap point, consider it as on idead match
    then remove the row and column of that cell, then repeat the process until
    the size of matrix is 1 (single cell)
    :param mat: the matrix to search
    :param rows: label/index for each row
    :param cols: label/index for each colm
    """
    n_row, n_col = mat.shape
    assert n_row == n_col, 'can only find permutation in square matrix'
    assert n_row == len(rows), 'Mismatch matrix size and row labels'
    assert n_col == len(cols), 'Mismatch matrix size and row labels'
    assert len(rows) == len(cols), 'Mismatch number of cols and rows labels'
    if n_row == 1 and n_col == 1:
        return [(rows[0], cols[0])]
    else:
        # Find the max
        max_row, max_col = np.unravel_index(mat.argmax(), mat.shape)
        row_label, col_label = rows[max_row], cols[max_col]
        reduce_mat = np.delete(mat, max_row, 0)
        reduce_mat = np.delete(reduce_mat, max_col, 1)
        r_rows = rows
        r_cols = cols
        del r_rows[max_row]
        del r_cols[max_col]
        # print '---> ',rows[max_row],cols[max_col]
        return [(row_label, col_label)] + find_best_row_permutation(reduce_mat, r_rows, r_cols)


if __name__ == '__main__':
    # Read the training data
    doc_ids = [line.rstrip().decode('utf8').encode('utf8') for line in open('./data/nyt.online_lead_top10.shelf_2001_test.ids')]
    multi_docs = [line.rstrip().decode('utf8').encode('utf8') for line in open('./data/nyt.online_lead_top10.shelf_2001_test.text')]
    ground_truth = [line.rstrip().decode('utf8').encode('utf8') for line in open('./data/nyt.online_lead_top10.shelf_2001_test.truth')]
    train_multi_docs = [line.rstrip().decode('utf8').encode('utf8') for line in open('./data/nyt.online_lead_top10.shelf_2001_train.text')]
    train_ground_truth = [line.rstrip().decode('utf8').encode('utf8') for line in open('./data/nyt.online_lead_top10.shelf_2001_train.truth')]
    cluster_overlap = 'count'
    # loop through document list
    for i in train_multi_docs:
        
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        tokens = [unicode(tok, errors='replace') for tok in tokens]
    
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if i not in en_stop]
        
        # stem tokens
        stemmed_tokens = [lemmatizer.lemmatize(i) for i in stopped_tokens]
        
        # add tokens to list
        texts.append(stemmed_tokens)
    
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
        
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    #  #########  CONVENTIONAL LDA
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=20)
    # ldamodel_mallet = gensim.models.wrappers.LdaMallet('../mallet-2.0.8/bin/mallet', corpus=corpus, num_topics=n_topics, id2word=dictionary)
    lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=n_topics)
    # START THE EVALUATION HERE
    predict = Predict(dictionary=dictionary, model=ldamodel)
    # predict_mallet = Predict(dictionary=dictionary, model=ldamodel_mallet)
    predict_lsi = Predict(dictionary=dictionary, model=lsi)
    assert ground_truth, 'Invalid ground-truth for evaluating model'
    assert len(ground_truth) == len(multi_docs), 'Ground-truth and document content have different size'

    result = dict()
    # result_mallet = dict()
    result_lsi = dict()
    for i in xrange(0,len(multi_docs)):
        doc_id = doc_ids[i] if doc_ids else None
        content = multi_docs[i]
        try:
            rs = predict.run(multi_docs[i])
            # rs_mallet = predict_mallet.run(multi_docs[i])
            rs_lsi = predict_lsi.run(multi_docs[i])
            rs = sorted(rs, key=lambda x: x[1], reverse=True)
            # rs_mallet = sorted(rs_mallet, key=lambda x: x[1], reverse=True)
            rs_lsi = sorted(rs_lsi, key=lambda x: x[1], reverse=True)
            #print rs
            # print len(rs), len(rs_lsi)
            result[doc_ids[i]] = rs[0][0] if rs else np.random.randint(n_topics)
            # result_mallet[doc_ids[i]] = rs_mallet[0][0] if rs_mallet else np.random.randint(n_topics)
            result_lsi[doc_ids[i]] = rs_lsi[0][0] if rs_lsi else np.random.randint(n_topics)
        except Exception as ins:
            print ins

    predicted_result = result
    # predicted_result_mallet = result_mallet
    # print len(result)
    predicted_result_lsi = result_lsi
    # print len(result_lsi)
    # Now we really evaluate it
    # print "Predicted predicted_result: ", predicted_result
    # Now evaluate the result
    # First reassign generated to the ground truth value

    # First transform ground-truth to cluster - like
    truth_clusters = dict()
    # do similar for the ground truth
    for i in xrange(0, len(ground_truth)):
        label = ground_truth[i]
        if label not in truth_clusters:
            truth_clusters[label] = [doc_ids[i]]
        else:
            truth_clusters[label].append(doc_ids[i])

    # convert predicted result to a dictionary form {topic_label: [list of all docs assigned this label]}
    predicted_clusters = dict()
    # the cluster
    for doc_id in predicted_result:
        label = predicted_result[doc_id]
        # print "Predicted label: ", label
        if label not in predicted_clusters:
            predicted_clusters[label] = [doc_id]
        else:
            predicted_clusters[label].append(doc_id)
    # This is TRICKY, if the size of predicted and truth are different, the create dummy cluster names
    diff = len(predicted_clusters) - len(truth_clusters)
    for i in xrange(0, abs(diff)):
        if diff < 0:
            predicted_clusters['dummy' + str(i)] = []
        elif diff > 0:
            truth_clusters['dummy' + str(i)] = []

    # with GIB
    '''
    predicted_clusters_mallet = dict()
    # the cluster
    for doc_id in predicted_result_mallet:
        label = predicted_result_mallet[doc_id]
        # print "Predicted label: ", label
        if label not in predicted_clusters_mallet:
            predicted_clusters_mallet[label] = [doc_id]
        else:
            predicted_clusters_mallet[label].append(doc_id)
    # This is TRICKY, if the size of predicted and truth are different, the create dummy cluster names
    diff = len(predicted_clusters_mallet) - len(truth_clusters)
    for i in xrange(0, abs(diff)):
        if diff < 0:
            predicted_clusters_mallet['dummy' + str(i)] = []
        elif diff > 0:
            truth_clusters['dummy' + str(i)] = []
	'''
    predicted_clusters_lsi = dict()
    # the cluster
    for doc_id in predicted_result_lsi:
        label = predicted_result_lsi[doc_id]
        # print "Predicted label: ", label
        if label not in predicted_clusters_lsi:
            predicted_clusters_lsi[label] = [doc_id]
        else:
            predicted_clusters_lsi[label].append(doc_id)
    # This is TRICKY, if the size of predicted and truth are different, the create dummy cluster names
    diff = len(predicted_clusters_lsi) - len(truth_clusters)
    for i in xrange(0, abs(diff)):
        if diff < 0:
            predicted_clusters_lsi['dummy' + str(i)] = []
        elif diff > 0:
            truth_clusters['dummy' + str(i)] = []

    # make dictionaries
    truth_topic_names = truth_clusters.keys()
    truth_topic_dictionary = {k: v for k, v in enumerate(truth_topic_names)}  # id-> name
    inv_truth_topic_dictionary = {truth_topic_dictionary[k]: k for k in truth_topic_dictionary}
    predicted_topic_names = predicted_clusters.keys()
    # predicted_topic_names_mallet = predicted_clusters_mallet.keys()
    predicted_topic_names_lsi = predicted_clusters_lsi.keys()
    predicted_topic_dictionary = {k: v for k, v in enumerate(predicted_topic_names)}  # id -> name
    # predicted_topic_dictionary_mallet = {k: v for k, v in enumerate(predicted_topic_names_mallet)}  # id -> name
    predicted_topic_dictionary_lsi = {k: v for k, v in enumerate(predicted_topic_names_lsi)}  # id -> name
    inv_predicted_topic_dictionary = {predicted_topic_dictionary[k]: k for k in predicted_topic_dictionary}
    # inv_predicted_topic_dictionary_mallet = {predicted_topic_dictionary_mallet[k]: k for k in predicted_topic_dictionary_mallet}
    inv_predicted_topic_dictionary_lsi = {predicted_topic_dictionary_lsi[k]: k for k in predicted_topic_dictionary_lsi}
    print 'Done making dictionaries'

    y_predicted = [inv_predicted_topic_dictionary[predicted_result[doc_id]] for doc_id in doc_ids]
    # y_predicted_mallet = [inv_predicted_topic_dictionary_mallet[predicted_result_mallet[doc_id]] for doc_id in doc_ids]
    print 'Predicted', y_predicted
    y_predicted_lsi = [inv_predicted_topic_dictionary_lsi[predicted_result_lsi[doc_id]] for doc_id in doc_ids]
    print 'Predicted LSI', y_predicted_lsi
    y_truth = [inv_truth_topic_dictionary[label] for label in ground_truth]
    print 'Truth', y_truth
    rs = {
        'adjusted_rand_score': metrics.adjusted_rand_score(y_truth, y_predicted),
        'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(y_truth, y_predicted),
        'homogeneity_score': metrics.homogeneity_score(y_truth, y_predicted),
        'completeness_score': metrics.completeness_score(y_truth, y_predicted),
        'v_measure_score': metrics.v_measure_score(y_truth, y_predicted),
        'fowlkes_mallows_score': metrics.adjusted_rand_score(y_truth, y_predicted),
        'silhouette_score': metrics.adjusted_rand_score(y_truth, y_predicted),

    }

    print 'CLUSTER LDA', rs
    '''
    rs = {
        'adjusted_rand_score': metrics.adjusted_rand_score(y_truth, y_predicted_mallet),
        'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(y_truth, y_predicted_mallet),
        'homogeneity_score': metrics.homogeneity_score(y_truth, y_predicted_mallet),
        'completeness_score': metrics.completeness_score(y_truth, y_predicted_mallet),
        'v_measure_score': metrics.v_measure_score(y_truth, y_predicted_mallet),
        'fowlkes_mallows_score': metrics.adjusted_rand_score(y_truth, y_predicted_mallet),
        'silhouette_score': metrics.adjusted_rand_score(y_truth, y_predicted_mallet),

    }

    # maybe_print("Model evaluation result: {0}".format(rs))
    print 'CLUSTER MALLET', rs
    '''
    rs = {
        'adjusted_rand_score': metrics.adjusted_rand_score(y_truth, y_predicted_lsi),
        'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(y_truth, y_predicted_lsi),
        'homogeneity_score': metrics.homogeneity_score(y_truth, y_predicted_lsi),
        'completeness_score': metrics.completeness_score(y_truth, y_predicted_lsi),
        'v_measure_score': metrics.v_measure_score(y_truth, y_predicted_lsi),
        'fowlkes_mallows_score': metrics.adjusted_rand_score(y_truth, y_predicted_lsi),
        'silhouette_score': metrics.adjusted_rand_score(y_truth, y_predicted_lsi),

    }

    # maybe_print("Model evaluation result: {0}".format(rs))
    print 'CLUSTER LSI', rs

    # Error analysis --- COMPUTE OVERLAP MATRIX
    # LDA
    overlap_matrix = np.zeros([len(predicted_clusters), len(truth_clusters)], dtype=np.float)
    for i in xrange(0, len(predicted_topic_names)):
        gen_topic = predicted_topic_names[i]
        for j in xrange(0, len(truth_topic_names)):
            truth_topic = truth_topic_names[j]
            if cluster_overlap == 'count':
                # print '\n',predicted_clusters
                overlap_matrix[i][j] = len(set(predicted_clusters[gen_topic])
                                           .intersection(set(truth_clusters[truth_topic])))
            elif cluster_overlap == 'percentage':
                overlap_matrix[i][j] = len(set(predicted_clusters[gen_topic])
                                           .intersection(set(truth_clusters[truth_topic]))) \
                                       / (len(set(predicted_clusters[gen_topic]))
                                          + len(set(truth_clusters[truth_topic]))
                                          )
    # MALLET
    '''
    overlap_matrix_mallet = np.zeros([len(predicted_clusters_mallet), len(truth_clusters)], dtype=np.float)
    for i in xrange(0, len(predicted_topic_names_mallet)):
        gen_topic = predicted_topic_names_mallet[i]
        for j in xrange(0, len(truth_topic_names)):
            truth_topic = truth_topic_names[j]
            if cluster_overlap == 'count':
                # print '\n',predicted_clusters
                overlap_matrix_mallet[i][j] = len(set(predicted_clusters_mallet[gen_topic])
                                           .intersection(set(truth_clusters[truth_topic])))
            elif cluster_overlap == 'percentage':
                overlap_matrix_mallet[i][j] = len(set(predicted_clusters_mallet[gen_topic])
                                           .intersection(set(truth_clusters[truth_topic]))) \
                                       / (len(set(predicted_clusters_mallet[gen_topic]))
                                          + len(set(truth_clusters[truth_topic]))
                                          )
    '''
    # LSI
    overlap_matrix_lsi = np.zeros([len(predicted_clusters_lsi), len(truth_clusters)], dtype=np.float)
    for i in xrange(0, len(predicted_topic_names_lsi)):
        gen_topic = predicted_topic_names_lsi[i]
        for j in xrange(0, len(truth_topic_names)):
            truth_topic = truth_topic_names[j]
            if cluster_overlap == 'count':
                # print '\n',predicted_clusters
                overlap_matrix_lsi[i][j] = len(set(predicted_clusters_lsi[gen_topic])
                                           .intersection(set(truth_clusters[truth_topic])))
            elif cluster_overlap == 'percentage':
                overlap_matrix_lsi[i][j] = len(set(predicted_clusters_lsi[gen_topic])
                                           .intersection(set(truth_clusters[truth_topic]))) \
                                       / (len(set(predicted_clusters_lsi[gen_topic]))
                                          + len(set(truth_clusters[truth_topic]))
                                          )

    # permutation selection
    # LDA
    tmp = find_best_row_permutation(mat=overlap_matrix,
                                         rows=list(xrange(0, len(predicted_topic_names))),
                                         cols=list(xrange(0, len(truth_topic_names))))
    best_indices = [i for i, _ in sorted(tmp, key=lambda x: x[1])]
    best_permutation = [predicted_topic_names[i] for i in best_indices]
    best_sum_score = np.sum(overlap_matrix[best_indices, list(xrange(0, len(truth_topic_names)))])

    # MALLET
    '''
    print overlap_matrix_mallet.shape
    tmp = find_best_row_permutation(mat=overlap_matrix_mallet,
                                         rows=list(xrange(0, len(predicted_topic_names_mallet))),
                                         cols=list(xrange(0, len(truth_topic_names))))
    best_indices_mallet = [i for i, _ in sorted(tmp, key=lambda x: x[1])]
    best_permutation_mallet = [predicted_topic_names_mallet[i] for i in best_indices_mallet]
    best_sum_score_mallet = np.sum(overlap_matrix_mallet[best_indices_mallet, list(xrange(0, len(truth_topic_names)))])

    '''
    # LSI
    tmp = find_best_row_permutation(mat=overlap_matrix_lsi,
                                         rows=list(xrange(0, len(predicted_topic_names_lsi))),
                                         cols=list(xrange(0, len(truth_topic_names))))
    best_indices_lsi = [i for i, _ in sorted(tmp, key=lambda x: x[1])]
    best_permutation_lsi = [predicted_topic_names_lsi[i] for i in best_indices_lsi]
    best_sum_score_lsi = np.sum(overlap_matrix_lsi[best_indices_lsi, list(xrange(0, len(truth_topic_names)))])

    # make dictionary
    # LDA, MALLET, LDA all in one
    predicted_topic_dictionary = dict()
    # predicted_topic_dictionary_mallet = dict()
    predicted_topic_dictionary_lsi = dict()
    for key in truth_topic_dictionary:
        predicted_topic_dictionary[key] = best_permutation[truth_topic_names.index(truth_topic_dictionary[key])]
        # predicted_topic_dictionary_mallet[key] = best_permutation_mallet[truth_topic_names.index(truth_topic_dictionary[key])]
        predicted_topic_dictionary_lsi[key] = best_permutation_lsi[truth_topic_names.index(truth_topic_dictionary[key])]

    inv_predicted_topic_dictionary = {predicted_topic_dictionary[k]: k for k in predicted_topic_dictionary}
    # inv_predicted_topic_dictionary_mallet = {predicted_topic_dictionary_mallet[k]: k for k in predicted_topic_dictionary_mallet}
    inv_predicted_topic_dictionary_lsi = {predicted_topic_dictionary_lsi[k]: k for k in predicted_topic_dictionary_lsi}
    y_predicted = [inv_predicted_topic_dictionary[predicted_result[doc_id]] for doc_id in doc_ids]
    # y_predicted_mallet = [inv_predicted_topic_dictionary_mallet[predicted_result_mallet[doc_id]] for doc_id in doc_ids]
    y_predicted_lsi = [inv_predicted_topic_dictionary_lsi[predicted_result_lsi[doc_id]] for doc_id in doc_ids]

    # LDA
    rs_error = {
        'accuracy': metrics.accuracy_score(y_truth, y_predicted),
        'precision_macro': metrics.precision_score(y_truth, y_predicted, average='macro'),
        'precision_micro': metrics.precision_score(y_truth, y_predicted, average='micro'),
        'precision_weighted': metrics.precision_score(y_truth, y_predicted, average='weighted'),
        # 'average_precision': metrics.average_precision_score(y_truth, y_predicted),
        'recall_macro': metrics.recall_score(y_truth, y_predicted, average='macro'),
        'recall_micro': metrics.recall_score(y_truth, y_predicted, average='micro'),
        'recall_weighted': metrics.recall_score(y_truth, y_predicted, average='weighted'),
        'f1_macro': metrics.f1_score(y_truth, y_predicted, average='macro'),
        'f1_micro': metrics.f1_score(y_truth, y_predicted, average='micro'),
        'f1_weighted': metrics.f1_score(y_truth, y_predicted, average='weighted'),
    }
    print "ERR LDA: ",rs_error

    # MALLET
    '''
    rs_error = {
        'accuracy': metrics.accuracy_score(y_truth, y_predicted_mallet),
        'precision_macro': metrics.precision_score(y_truth, y_predicted_mallet, average='macro'),
        'precision_micro': metrics.precision_score(y_truth, y_predicted_mallet, average='micro'),
        'precision_weighted': metrics.precision_score(y_truth, y_predicted_mallet, average='weighted'),
        # 'average_precision': metrics.average_precision_score(y_truth, y_predicted),
        'recall_macro': metrics.recall_score(y_truth, y_predicted_mallet, average='macro'),
        'recall_micro': metrics.recall_score(y_truth, y_predicted_mallet, average='micro'),
        'recall_weighted': metrics.recall_score(y_truth, y_predicted_mallet, average='weighted'),
        'f1_macro': metrics.f1_score(y_truth, y_predicted_mallet, average='macro'),
        'f1_micro': metrics.f1_score(y_truth, y_predicted_mallet, average='micro'),
        'f1_weighted': metrics.f1_score(y_truth, y_predicted_mallet, average='weighted'),
    }
    print "ERR MALLET: ",rs_error
	'''
    # LSI
    rs_error = {
        'accuracy': metrics.accuracy_score(y_truth, y_predicted_lsi),
        'precision_macro': metrics.precision_score(y_truth, y_predicted_lsi, average='macro'),
        'precision_micro': metrics.precision_score(y_truth, y_predicted_lsi, average='micro'),
        'precision_weighted': metrics.precision_score(y_truth, y_predicted_lsi, average='weighted'),
        # 'average_precision': metrics.average_precision_score(y_truth, y_predicted),
        'recall_macro': metrics.recall_score(y_truth, y_predicted_lsi, average='macro'),
        'recall_micro': metrics.recall_score(y_truth, y_predicted_lsi, average='micro'),
        'recall_weighted': metrics.recall_score(y_truth, y_predicted_lsi, average='weighted'),
        'f1_macro': metrics.f1_score(y_truth, y_predicted_lsi, average='macro'),
        'f1_micro': metrics.f1_score(y_truth, y_predicted_lsi, average='micro'),
        'f1_weighted': metrics.f1_score(y_truth, y_predicted_lsi, average='weighted'),
    }
    print "ERR LSI: ",rs_error
