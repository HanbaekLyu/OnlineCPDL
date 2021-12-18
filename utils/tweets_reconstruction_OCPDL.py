from utils.ocpdl import Online_CPDL
# from utils import plotting
# from utils import utils

import numpy as np
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from skimage.transform import downscale_local_mean
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition import SparseCoder
from time import time
import itertools
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["axes.edgecolor"] = "0.6"
plt.rcParams["axes.labelsize"] = 26
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "serif"
plt.rcParams["grid.color"] = "0.85"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["legend.columnspacing"] *= 0.8
plt.rcParams["legend.edgecolor"] = "0.6"
plt.rcParams["legend.markerscale"] = 1.0
plt.rcParams["legend.framealpha"] = "1"
plt.rcParams["legend.handlelength"] *= 1.5
plt.rcParams["legend.numpoints"] = 2
plt.rcParams["text.usetex"] = True
plt.rcParams["xtick.major.pad"] = -3
plt.rcParams["ytick.major.pad"] = -2
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["figure.figsize"] = [12.0, 6.0]


import matplotlib.font_manager as font_manager
import matplotlib.gridspec as gridspec
import matplotlib
import os
import re
import psutil
import pickle
import pandas
from collections import defaultdict
# Date times
import datetime
import os
import pickle
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns

DEBUG = False


class Tweets_Reconstructor_OCPDL():
    ### Use Online CP Dictionary Learning for patch-based image processing
    def __init__(self,
                 path,
                 n_components=100,  # number of dictionary elements -- rank
                 iterations=50,  # number of iterations for the ONTF algorithm
                 sub_iterations=20,  # number of i.i.d. subsampling for each iteration of ONTF
                 batch_size=20,  # number of patches used in i.i.d. subsampling
                 num_patches=1000,  # number of patches that ONTF algorithm learns from at each iteration
                 sub_num_patches=10000,  # number of patches to optimize H after training W
                 segment_length=20,  # number of video frames in a single video patch
                 unfold_words_tweets=False,
                 if_sample_from_tweets_mode=True,
                 alpha=1):
        '''
        batch_size = number of patches used for training dictionaries per ONTF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.path = path
        self.n_components = n_components
        self.iterations = iterations
        self.sub_iterations = sub_iterations
        self.num_patches = num_patches
        self.sub_num_patches = sub_num_patches
        self.alpha = alpha
        self.batch_size = batch_size
        self.unfold_words_tweets = unfold_words_tweets
        self.if_sample_from_tweets_mode = if_sample_from_tweets_mode
        self.segment_length = segment_length
        self.W = np.zeros(shape=(1, n_components))  ### Will be re-initialized by ocpdl.py with the correct shape
        self.code = np.zeros(shape=(n_components, iterations * batch_size))
        self.sequential_dict = {}

        dict = pickle.load(open(self.path, "rb"))
        self.X_words = dict[0]  ### list of words
        # self.X_retweetcounts = dict[1]  ### retweet counts
        self.data = dict[1] * 10000 ### [timeframes, words, tweets]  ### use dict[2] for old tweet data
        ### scale by 100 since the original tfidf weigts are too small -- result of learning is noisy
        print('data_shape ([timeframes, words, tweets])', self.data.shape)

        self.frameCount = self.data.shape[0]
        self.num_words = self.data.shape[1]
        self.num_tweets = self.data.shape[2]

    def extract_random_patches(self, data=None, starting_time=None, if_sample_from_tweets_mode=True):
        '''
        Extract 'num_patches' many random patches of given size
        Two tensor data types depending on how to unfold (seg_length x num_words x num_tweets) short time-series patches:
            unfold_words_tweets : (seg_length, num_words * num_tweets, 1 , 1)
            else: (seg_length, num_words,  num_tweets, 1)
            last mode = batch mode
        '''
        kt = self.frameCount
        ell = self.segment_length
        num_patches = self.num_patches

        if not if_sample_from_tweets_mode:
            if self.unfold_words_tweets:
                X = np.zeros(shape=(ell, self.num_words * self.num_tweets, 1, 1))
            else:
                X = np.zeros(shape=(ell, self.num_words, self.num_tweets, 1))

            for i in np.arange(num_patches):
                if starting_time is None:
                    a = np.random.choice(kt - ell)  # starting time of the short time-series patch
                else:
                    a = starting_time

                if data is None:
                    Y = self.data[a:a + ell, :, :]  # ell by num_words by num_tweets
                else:
                    Y = data[a:a + ell, :, :]  # ell by num_words by num_tweets
                # print('Y.shape', Y.shape)

                if self.unfold_words_tweets:
                    Y = Y.reshape(ell, self.num_words * self.num_tweets, 1, 1)
                else:
                    Y = Y.reshape(ell, self.num_words, self.num_tweets, 1)

                if i == 0:
                    X = Y
                else:
                    X = np.append(X, Y, axis=3)  # X is class ndarray
        else:
            ### Sample patches from the tweets mode instead of the time mode
            ### ell = self.segment_length plays the fole of patch size
            for i in np.arange(num_patches):
                a = np.arange(self.num_tweets)
                idx = np.random.choice(a, ell)

                if data is None:
                    Y = self.data[:, :, idx]  # ell by num_words by num_tweets
                else:
                    Y = data[:, :, idx]  # ell by num_words by num_tweets
                # print('Y.shape', Y.shape)
                Y = Y.reshape(data.shape[0], data.shape[1], ell, 1)

                if i == 0:
                    X = Y
                else:
                    X = np.append(X, Y, axis=3)  # X is class ndarray

        return X

    def out(self, loading):
        ### given loading, take outer product of respected columns to get CPdict
        CPdict = {}
        for i in np.arange(self.n_components):
            A = np.array([1])
            for j in np.arange(len(loading.keys())):
                loading_factor = loading.get('U' + str(j))  ### I_i X n_components matrix
                # print('loading_factor', loading_factor)
                A = np.multiply.outer(A, loading_factor[:, i])
            A = A[0]
            CPdict.update({'A' + str(i): A})
        return CPdict

    def grey_color_func(self, word, font_size, position, orientation, random_state=None,
                        **kwargs):
        return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 100)

    def random_color_func(self, word=None, font_size=None, position=None, orientation=None, font_path=None,
                          random_state=None):
        h = int(360.0 * 21.0 / 255.0)
        s = int(100.0 * 255.0 / 255.0)
        l = int(100.0 * float(random_state.randint(180, 255)) / 255.0)

        return "hsl({}, {}%, {}%)".format(h, s, l)

    def display_dictionary_CP(self,
                              W,
                              save_fig_name=None,
                              num_word_sampling_from_topics=100,
                              num_top_words_from_each_topic=10,
                              if_plot=False):

        num_rows = np.floor(np.sqrt(self.n_components)).astype(int)
        num_cols = np.ceil(np.sqrt(self.n_components)).astype(int)

        U0 = W.get('U0')  ### dict for time mode
        U1 = W.get('U1')  ### dict for words mode (topics)
        U2 = W.get('U2')  ### dict for tweets mode

        ### topic mode
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8, 6),
                                subplot_kw={'xticks': [], 'yticks': []})
        # fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(6, 4.2),
        #                          subplot_kw={'xticks': [], 'yticks': []})
        for ax, i in zip(axs.flat, range(self.n_components)):
            print('Sampling from the %i th topic' % i)
            patch = U1[:, i]
            I = np.argsort(patch)
            I = np.flip(I)
            # print('patch[I]', patch[I])
            I = I[0:num_top_words_from_each_topic]
            # print('I', I)
            patch_reduced = patch[I]
            dist = patch_reduced.copy() / np.sum(
                patch_reduced)  ### probability distribution on the words given by the ith topic vector

            ### Randomly sample a word from the corpus according to the PMF "dist" multiple times
            ### to generate text data corresponding to the ith topic, and then generate its wordcloud
            list_words = []
            a = np.arange(len(patch_reduced))

            for j in range(num_word_sampling_from_topics):
                idx = np.random.choice(a, p=dist)
                list_words.append(self.X_words[I[idx]])
                # print('self.X_words[idx]', self.X_words[idx])
            # print(list_words)
            Y = " ".join(list_words)
            stopwords = STOPWORDS
            stopwords.update(["’", "“", "”", "000", "000 000", "https", "co", "19", "2019", "coronavirus",
                              "virus", "corona", "covid", "ncov", "covid19", "amp"])
            wordcloud = WordCloud(stopwords=stopwords,
                                  background_color="black",
                                  relative_scaling=1,
                                  width=400,
                                  height=400).generate(Y)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")

        plt.tight_layout()
        plt.suptitle('Topics mode', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.08)
        plt.savefig('Tweets_dictionary/fig_topics' +'_'+ str(save_fig_name))
        if if_plot:
            plt.show()


        ### time mode
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6),
                                subplot_kw={'xticks': [], 'yticks': []})

        ### Normalize code matrix
        for i in np.arange(self.n_components):
            U0[:,i] /= np.sum(U0[:,i])
            print('np.sum(U0[:,i])', np.sum(U0[:,i]))

        axs.imshow(U0, cmap="viridis", interpolation='nearest', aspect='auto')
        print('time_mode_shape')
        print('time_mode:', U0[:, i].reshape(1, -1))

        plt.tight_layout()
        plt.suptitle('Time mode', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.00, 0.00)
        plt.savefig('Tweets_dictionary/fig_temporal_modes' +'_'+ str(save_fig_name))
        if if_plot:
            plt.show()

        '''
        ### time mode
        fig, axs = plt.subplots(nrows=self.n_components, ncols=1, figsize=(12, 4),
                                subplot_kw={'xticks': [], 'yticks': []})

        for ax, i in zip(axs.flat, range(self.n_components)):
            ax.imshow(U0[:, i].reshape(1, -1), cmap="gray_r", interpolation='nearest')
            print('time_mode_shape')
            print('time_mode:', U0[:, i].reshape(1, -1))

        plt.tight_layout()
        plt.suptitle('Time mode', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.08)
        plt.savefig('Tweets_dictionary/fig_temporal_modes' +'_'+ str(save_fig_name))
        if if_plot:
            plt.show()
        '''


    def display_topics_denali(self, W):

        U0 = W.get('U0')  ### dict for time mode
        U1 = W.get('U1')  ### dict for words mode (topics)
        U2 = W.get('U2')  ### dict for tweets mode

        topics_freqs = []
        for i, topic in enumerate(U1.T):
            topics_freqs.append({self.X_words[i]: topic[i] for i in reversed(topic.argsort()[-10:])})
            print("Topic {}: {}".format(i, ', '.join(x for x in reversed(np.array(self.X_words)[topic.argsort()[-10:]]))))

        topics_freqs = condense_topic_keywords(topics_freqs)
        num_keywords = 5

        # Make word and frequency lists for each topic.
        sorted_topics = [sorted(topic.items(), key=lambda item: item[1], reverse=True) for topic in topics_freqs]
        word_lists = [[item[0] for item in topic[:num_keywords]] for topic in sorted_topics]
        freq_lists = [[item[1] for item in topic[:num_keywords]] for topic in sorted_topics]

        # Plot topics barchart.
        plotting.plot_keyword_bars(freq_lists, word_lists, figsize=[20.0, 10.0])
        # plt.savefig('NMF_keyword_barplot_{}_{}_topics.png'.format(data_name, r))
        plt.show()


    def display_sequential_dictionary_CP(self,
                                         seq_dict,
                                         save_fig_name=None,
                                         num_word_sampling_from_topics=100,
                                         num_top_words_from_each_topic=10,
                                         list_of_topics2display = None,
                                         if_plot=False,
                                         if_sample_topic_words=False):
        ### Initial setup
        M = len(seq_dict.keys())

        W = seq_dict.get('W' + str(0))
        # U0 = W.get('U0')  ### dict for time mode
        # U1 = W.get('U1')  ### dict for words mode (topics)
        # U2 = W.get('U2')  ### dict for tweets mode

        nrows = M
        if list_of_topics2display is None:
            ncols = self.n_components
            list_of_topics2display = range(self.n_components)
        else:
            ncols = len(list_of_topics2display)

        ### Make gridspec
        fig1 = plt.figure(figsize=(15, 9), constrained_layout=False)
        # make outer gridspec


        outer_grid = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0.05, hspace=0.05)
        # make nested gridspecs
        for i in range(nrows * ncols):
            if i % ncols == 0:

                ### move to the next CP dictionary stored in seq_dict, which will be plotted in the next row
                W = seq_dict.get('W' + str(i  // ncols))  ### use floor division //
                U0 = W.get('U0')  ### dict for time mode
                U1 = W.get('U1')  ### dict for words mode (topics)
                U2 = W.get('U2')  ### dict for tweets mode

            inner_grid = outer_grid[i].subgridspec(1, ncols, wspace=0.00, hspace=0.05)
            # for j, (c, d) in enumerate(itertools.product(range(0,2), repeat=1)):

            for j in range(2):
                i = i % ncols
                i1 = list_of_topics2display[i]
                if j == 0:
                    ax = fig1.add_subplot(inner_grid[:, -1:])
                    b = U0[:, i1].reshape(1, -1).T
                    ax.imshow(b,
                              cmap="viridis",  ## black = 1, white = 0
                              interpolation='nearest', aspect='auto')
                    if i % ncols == 0:
                        print('U0[:,i]', U0[:, i1])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # box = ax.get_position()
                    # box.x0 = box.x0 - 1
                    # box.x1 = box.x1 - 1
                    # ax.set_position(box)
                else:
                    ax = fig1.add_subplot(inner_grid[:, :-1])
                    # print('Sampling from the %i th topic' % i)
                    patch = U1[:, i1]
                    # print('U1[:,i]', U1[:, i])
                    I = np.argsort(patch)
                    I = np.flip(I)
                    # print('patch[I]', patch[I])
                    I = I[0:num_top_words_from_each_topic]
                    # print('I', I)
                    patch_reduced = patch[I]
                    dist = patch_reduced.copy() / np.sum(
                        patch_reduced)  ### probability distribution on the words given by the ith topic vector


                    if if_sample_topic_words:
                        ### Randomly sample a word from the corpus according to the PMF "dist" multiple times
                        ### to generate text data corresponding to the ith topic, and then generate its wordcloud
                        list_words = []
                        a = np.arange(len(patch_reduced))

                        for j in range(num_word_sampling_from_topics):
                            idx = np.random.choice(a, p=dist)
                            list_words.append(self.X_words[I[idx]])
                            # print('self.X_words[idx]', self.X_words[idx])
                        # print(list_words)
                    else:
                        ### Simply take the top few topic words -- no relative weight info included
                        list_words = []
                        for j in range(num_top_words_from_each_topic):
                            list_words.append(self.X_words[I[j]])


                    Y = " ".join(list_words)
                    stopwords = STOPWORDS
                    stopwords.update(["’", "“", "”", "000", "000 000", "https", "co", "19", "2019", "coronavirus",
                                      "virus", "corona", "covid", "ncov", "covid19", "amp"])
                    wordcloud = WordCloud(stopwords=stopwords,
                                          background_color="black",
                                          relative_scaling=0,
                                          prefer_horizontal=1,
                                          width=400,
                                          height=400).generate(Y)
                    # ax.imshow(wordcloud, interpolation='bilinear')
                    ax.imshow(wordcloud.recolor(color_func=self.random_color_func, random_state=3),
                              interpolation="bilinear", aspect="auto")
                    ax.axis("off")
                    # ax.set_aspect('equal')

        plt.tight_layout()
        # plt.suptitle('Topics mode', fontsize=16)
        plt.subplots_adjust(0.00, 0.00, 1, 1, wspace=0.0, hspace=0.08)
        plt.savefig('Tweets_dictionary/fig_sequential_topics' + '_' + str(save_fig_name) + '.png')
        if if_plot:
            plt.show()

        '''
        ### color mode
        fig, axs = plt.subplots(nrows=self.n_components, ncols=1, figsize=(6, 6),
                                subplot_kw={'xticks': [], 'yticks': []})

        for ax, i in zip(axs.flat, range(self.n_components)):
            ax.imshow(U0[:, i].reshape(1, -1), cmap="gray", interpolation='nearest')

        plt.tight_layout()
        plt.suptitle('Time mode', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.08)
        plt.savefig('Tweets_dictionary/fig_temporal_modes')
        if if_plot:
            plt.show()
        '''
    def show_array(self, arr):
        plt.figure()
        plt.imshow(arr, cmap="gray")
        plt.show()

    def train_dict(self,
                   data=None,
                   ini_dict=None,
                   ini_At=None,
                   ini_Bt=None,
                   ini_history=None,
                   iterations=None,
                   sub_iterations=None,
                   save_file_name=0,
                   if_sample_from_tweets_mode=True,
                   if_save=False,
                   beta=None):
        print('training CP dictionaries from patches...')
        '''
        Trains dictionary based on patches from an i.i.d. sequence of batch of patches
        CP dictionary learning
        '''
        if ini_dict is not None:
            W = ini_dict ### Damn!! Don't make this None it will reset the seq_learning function
        else:
            W = None

        if ini_At is None:
            At = None
        else:
            At = ini_At

        if ini_Bt is None:
            Bt = None
        else:
            Bt = ini_Bt

        if ini_history is None:
            ini_his = 0
        else:
            ini_his = ini_history

        if iterations is None:
            iter = self.iterations
        else:
            iter = iterations

        if sub_iterations is None:
            sub_iter = self.sub_iterations
        else:
            sub_iter = sub_iterations

        for t in np.arange(iter):
            if data is None:
                X = self.extract_random_patches(data=self.data,
                                                if_sample_from_tweets_mode=if_sample_from_tweets_mode)
            else:
                X = self.extract_random_patches(data=data,
                                                if_sample_from_tweets_mode=if_sample_from_tweets_mode)

            # print('X.shape', X.shape)

            if t == 0:
                self.ntf = Online_CPDL(X,
                                       self.n_components,
                                       ini_loading=W,
                                       ini_A = At,
                                       ini_B = Bt,
                                       iterations=sub_iter,
                                       batch_size=self.batch_size,
                                       alpha=self.alpha,
                                       history=ini_his,
                                       subsample=False,
                                       beta=beta)
                W, At, Bt, H = self.ntf.train_dict(output_results=False)

                print('in training: ini_his = ', ini_his)

            else:
                self.ntf = Online_CPDL(X, self.n_components,
                                       iterations=sub_iter,
                                       batch_size=self.batch_size,
                                       ini_loading=W,
                                       ini_A=At,
                                       ini_B=Bt,
                                       alpha=self.alpha,
                                       history=ini_his + t,
                                       subsample=False,
                                       beta=beta)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                W, At, Bt, H = self.ntf.train_dict(output_results=False)

                # code += H
            print('Current iteration %i out of %i' % (t, iter))
        self.W = self.ntf.loading
        # print('dict_shape:', self.W.shape)
        # print('code_shape:', self.code.shape)

        if if_save:
            np.save('Tweets_dictionary/dict_learned_CPDL_' + str(save_file_name), self.W)
            np.save('Tweets_dictionary/code_learned_CPDL_' + str(save_file_name), self.code)
        return W, At, Bt, H

    def train_sequential_dict(self,
                              save_file_name=0,
                              slide_window_by=10,
                              refresh_history_as=None,
                              beta=None):
        print('Sequentially training CP dictionaries from patches...')
        '''
        Trains dictionary based on patches from a sequence of batch of patches
        slide the window only forward
        if refresh_history=True, aggregation matrices are reset every iteration
        CP dictionary learning
        '''
        W = self.W
        At = None
        Bt = None
        t0 = 0
        t = 0 ### If history starts at 0, then t^{-\beta} goes from 0 to 1 to very small.
              ### To ensure monotonicity, start at t=1.
        seq_dict = {}

        while t0 + t * slide_window_by + self.segment_length <= self.frameCount:
            # X = self.extract_random_patches(if_sample_from_tweets_mode=False)
            X = self.extract_random_patches(starting_time=t0 + t * slide_window_by,
                                            if_sample_from_tweets_mode=False)

            # print('X', X)
            print('X.shape', X.shape)
            t1 = t0 + t * slide_window_by
            print('Current time slab starts at %i' % t1)

            if refresh_history_as is not None:
                ini_history = refresh_history_as
            else:
                ini_history = (t+1)*self.iterations  ### To ensure monotonicity, start at t=1.

            ### data tensor still too large -- subsample from the tweets mode:

            if t == 0:
                ini_dict = None
            else:
                ini_dict = W

            print('ini_history', ini_history)

            if t > 0:
                print('U0 right before new phase', ini_dict.get('U0')[0,0])


            W, At, Bt, H = self.train_dict(data=X,
                                           ini_dict=ini_dict,
                                           ini_At = At,
                                           ini_Bt = Bt,
                                           iterations=self.iterations,
                                           sub_iterations=2,
                                           ini_history=ini_history,
                                           if_sample_from_tweets_mode=True,
                                           if_save=False,
                                           beta=beta)

            seq_dict.update({'W' + str(t): W.copy()})

            # for i in np.arange(t):
            #    print('U0[:,0]', seq_dict.get('W' + str(i)).get('U0')[0,0])

            ### print out current memory usage
            pid = os.getpid()
            py = psutil.Process(pid)
            memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
            # print('memory use:', memoryUse)

            t += 1
            # print('Current iteration %i out of %i' % (t, self.iterations))


        # self.W = self.ntf.loading
        self.sequential_dict = seq_dict

        # print('dict_shape:', self.W.shape)
        # print('code_shape:', self.code.shape)
        # np.save('Tweets_dictionary/dict_learned_CPDL_'+str(save_file_name), self.W)
        np.save('Tweets_dictionary/sequential_dict_learned_CPDL_' + str(save_file_name), seq_dict)
        # np.save('Tweets_dictionary/code_learned_CPDL_'+str(save_file_name), self.code)

        return seq_dict


### From utils function in the team repo

def condense_topic_keywords(topics_freqs, num_keywords=5):
    """Remove bigram parts from top keywords by updating frequencies.

    Args:
        topics_freqs (list of dictionaries):
            Association strength between term and topic.

    Returns:
        topics_freqs (list of dictionaries)

    Example:
    >>> words = [["a", "b", "a b"],["b", "a c", "c"], ["a", "b", "c"]]
    >>> test = [{word_list[i]: i + 1 for i in range(3)} for word_list in words]
    >>> condense_topic_keywords(test)
    [{'a': 0, 'b': 0, 'a b': 3}, {'b': 1, 'a c': 3, 'c': 0}, {'a': 1, 'b': 2, 'c': 3}]
    """
    for k, topic_freqs in enumerate(topics_freqs):
        bigrams = defaultdict(list)
        unigrams = set()
        for (term, freqs) in topic_freqs.items():
            # Drop terms which have no english letters.
            if not re.match(r".*[a-zA-Z].*", term):
                topic_freqs[term] = 0
                continue

            # Form dictionary of words to bigrams and a set of unigrams.
            term_words = term.split()
            if len(term_words) > 1:
                for sub_term in term_words:
                    bigrams[sub_term].append(term)
            else:
                unigrams.add(term)

        # Replace unigrams with bigrams and assign max weight of unigram and bigram.
        for term in unigrams:
            if term in bigrams:
                for bigram in bigrams[term]:
                    topic_freqs[bigram] = max(topic_freqs[bigram], topic_freqs[term])
                topic_freqs[term] = 0

    return topics_freqs



def heatmap(
    data,
    x_tick_labels=None,
    x_label="",
    y_tick_labels=None,
    y_label="",
    figsize=(8, 10),
):
    """Plot heatmap.

    Args:
        data: (2d array) data to be plotted (topics x date)
        x_tick_labels (list of str)

    Returns:
        fig
        ax
    """
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(data, cbar_kws = dict(use_gridspec=False,location="top"))

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)


    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(np.arange(0, data.shape[0], 1.0) + 0.5, fontsize=22)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)

    if y_tick_labels is None:
        labels = [topic_num + 1 for topic_num in range(data.shape[0])]
    ax.set_yticklabels(y_tick_labels)
    plt.yticks(rotation=0)

    if x_tick_labels is not None:
        labels = [x_tick_labels[int(item.get_text())] for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

    return fig, ax

def save_figure(fig, filepath=None, bbox_inches="tight", pad_inches=0.1):
    """Save figure in filepath."""
    if filepath is None:
        raise Exception("Filepath must be specified if save_fig=True.")
    fig_kwargs = {"bbox_inches": "tight", "pad_inches": 0.1, "transparent": True}
    fig.savefig(filepath + ".png", **fig_kwargs)
    fig.savefig(filepath + ".pdf", **fig_kwargs)
    # plt.close()

def shaded_latex_topics(freq_lists, word_lists, min_shade=0.6):
    """Generate latex table with keywords shaded by frequency value.
    Based on stackoverflow.com/questions/59220488/to-visualize-attention-color-tokens-using-attention-weights

    Args:
        freq_lists (list of lists of floats):
        word_lists (list of lists of strings):
        min_shade (float): min shade to be printed so that values are visible.

    Returns:
    (string) Latex table when printed.
    """

    if len(freq_lists) != len(word_lists):
        raise Exception("Frequency and word lists must have the same length.")

    if len(word_lists[0]) == 0:
        raise Exception("Word list is empty.")

    cmap = matplotlib.cm.Greys
    colored_string = "\\begin{{tabular}}{{|l{}|}} \\hline \n".format(
        "".join(["c" for i in range(len(word_lists[0]))])
    )

    for i, (freqs, words) in enumerate(zip(freq_lists, word_lists)):
        colored_string += "Topic {}: &".format(i + 1)

        # Normalize topic frequencies.
        freqs = np.array(freqs) / sum(freqs)

        # Get list of color shades.
        colors = [
            matplotlib.colors.rgb2hex(cmap(shifted_freqs))[1:]
            for shifted_freqs in (1 - min_shade) * freqs + min_shade
        ]

        # Add topic keywords to table.
        colored_string += "&".join(
            [
                "\\textcolor[HTML]{"
                + color
                + "}{"
                + "{} ({:.2f})".format(word, freq)
                + "} "
                for freq, word, color in zip(freqs, words, colors)
            ]
        )
        colored_string += "\\\\ \\hline \n"

    colored_string += "\\end{tabular}"
    return colored_string


def display_heatmap_topics(W, X_words, savepath):
    # We consider a TFIDF weight tensor of dimension timeframes-by-features-by-documents
    # - features (list): list of all features (words) extracted from documents
    # - retweet_counts (list): list containing the number of retweets for each corresponding tweet in corpus.
    # - X (ndarray): TFIDF weight tensor of dimension timeframes-by-features-by-documents

    U0 = W.get('U0')  ### dict for time mode
    U1 = W.get('U1')  ### dict for words mode (topics)
    U2 = W.get('U2')  ### dict for tweets mode

    r = U0.shape[1]
    #data_name = "top1000daily"
    #folder_name = "Tweets_dictionary"

    # Condense topic representations.
    topics_freqs = []
    for i, topic in enumerate(U1.T):
        topics_freqs.append({X_words[i]: topic[i] for i in reversed(topic.argsort()[-r:])})
        print(
            "Topic {}: {}".format(i, ', '.join(x for x in reversed(np.array(X_words)[topic.argsort()[-10:]]))))

    print('topics_freqs', topics_freqs.itmes())

    topics_freqs = condense_topic_keywords(topics_freqs)
    num_keywords = 5

    sns.set(style="whitegrid", context="talk")


    # Make word and frequency lists for each topic.
    sorted_topics = [
        sorted(topic.items(), key=lambda item: item[1], reverse=True)
        for topic in topics_freqs
    ]
    word_lists = [[item[0] for item in topic[:num_keywords]] for topic in sorted_topics]
    freq_lists = [[item[1] for item in topic[:num_keywords]] for topic in sorted_topics]
    # Print latex table with shaded topics.
    #table_filename = "NMF_{}_topic_keywords_{}_{}_topics.tex".format(
    #    folder_name, data_name, r
    #)
    # table_filepath = os.path.join(overleaf_dir, "Tables", table_filename)
    # with open(table_filepath, "w") as file:
    #   file.write(plotting.shaded_latex_topics(freq_lists, word_lists))
    # topic_latex = shaded_latex_topics(freq_lists, word_lists, min_shade=0.6)
    # print('!!! topic_latex', topic_latex)

    num_time_slices = U0.shape[0]

    avg_topics_over_time = U0.T
    for i in np.arange(U0.shape[0]):
        avg_topics_over_time[:,i] = avg_topics_over_time[:,i]/np.sum(avg_topics_over_time[:,i])

    # #### Visualize topic distributions
    start = datetime.datetime(2020, 2, 1, 0)
    dates = [i * datetime.timedelta(days=1) + start for i in range(num_time_slices)]
    date_strs = [date.strftime("%m-%d") for date in dates]
    # y_tick_labels = ["{}: {}".format(word_lists[i][0:2], i + 1) for i in range(r)]
    y_tick_labels = [str(word_lists[i][0])+ ", " +str(word_lists[i][1]) + ", "  +str(word_lists[i][2])  +" "+ str(i+1) for i in range(r)]

    fig, ax = heatmap(
        avg_topics_over_time,
        x_tick_labels=date_strs,
        x_label="Date",
        y_tick_labels=y_tick_labels,
        y_label="Topic",
    )
    # Save figure.
    fig_filename = "NMF_tweet_representation_of_topics_{}_{}_topics".format(
        data_name, r
    )
    # fig_filepath = "Tweets_dictionary/" + fig_filename
    fig_filepath = savepath
    save_figure(fig, filepath=fig_filepath)
    plt.show()
