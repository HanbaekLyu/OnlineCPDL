from utils.ocpdl import Online_CPDL
import numpy as np
from PIL import Image
from skimage.transform import downscale_local_mean
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition import SparseCoder
from time import time
import matplotlib.gridspec as gridspec
import itertools
import matplotlib.pyplot as plt

import cv2


DEBUG = False


class Video_Reconstructor_OCPDL():
    ### Use Online CP Dictionary Learning for patch-based image processing
    def __init__(self,
                 path,
                 n_components=100,  # number of dictionary elements -- rank
                 iterations=50,  # number of iterations for the ONTF algorithm
                 sub_iterations = 20,  # number of i.i.d. subsampling for each iteration of ONTF
                 batch_size=20,   # number of patches used in i.i.d. subsampling
                 num_patches = 1000,   # number of patches that ONTF algorithm learns from at each iteration
                 sub_num_patches = 10000,  # number of patches to optimize H after training W
                 downscale_factor=2,
                 patch_size=7,
                 segment_length=20, # number of video frames in a single video patch
                 patches_file='',
                 learn_joint_dict=False,
                 is_matrix=False,
                 unfold_space=False,
                 unfold_spaceNcolor=False,
                 unfold_all=False,
                 is_color=True):
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
        self.batch_size = batch_size
        self.downscale_factor = downscale_factor
        self.patch_size = patch_size
        self.patches_file = patches_file
        self.learn_joint_dict = learn_joint_dict
        self.is_matrix = is_matrix
        self.unfold_space = unfold_space
        self.unfold_all = unfold_all
        self.unfold_spaceNcolor=unfold_spaceNcolor
        self.is_color = is_color
        self.segment_length = segment_length
        self.W = np.zeros(shape=(patch_size, n_components))
        self.code = np.zeros(shape=(n_components, iterations*batch_size))

        # read in image as array
        if self.is_matrix:
            self.data = np.load(path)
            print('data.shape', self.data.shape)
        else:
            self.data = self.read_video_as_array()

        self.frameHeight = self.data.shape[1]
        self.frameWidth = self.data.shape[2]
        self.frameCount = self.data.shape[0]

    def read_video_as_array(self):
        cap = cv2.VideoCapture(self.path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

        fc = 0
        ret = True

        while (fc < frameCount and ret):
            ret, buf[fc] = cap.read()
            fc += 1

        cap.release()
        # cv2.namedWindow('frame 10')
        # cv2.imshow('frame 10', buf[9])
        # cv2.waitKey(0)

        print('data.shape', buf.shape)  ### frames * width * height * 3
        return buf

    def extract_random_patches(self):
        '''
        Extract 'num_patches' many random patches of given size
        Three tensor data types depending on how to unfold (seg_length by k by k by 3) color video patches:
            unfold_space : (seg_length, kx * ky, 3, 1)
            unfold_all : (seg_length, 3 * kx * ky, 1, 1)
            else: (seg_length, kx, ky, 3, 1)
        '''
        k = self.patch_size
        kx = self.frameWidth
        ky = self.frameHeight
        kt = self.frameCount
        ell = self.segment_length
        num_patches = self.num_patches

        if self.unfold_all:
            X = np.zeros(shape=(ell, 3 * kx * ky, 1, 1))
        elif self.unfold_space:
            X = np.zeros(shape=(ell, kx * ky, 3, 1))
        else:
            X = np.zeros(shape=(ell, kx, ky, 3, 1))

        for i in np.arange(num_patches):
            a = np.random.choice(kt - ell)  # starting time of the video patch
            # b = np.random.choice(x[1] - k)  # y coordinate of the top left corner of the random patch
            Y = self.data[a:a+ell, :, :, :] # ell by kx by ky by 3
            # print('Y.shape', Y.shape)

            if self.unfold_all:
                Y = Y.reshape(ell, 3 * kx * ky, 1, 1)

            elif self.unfold_space:
                Y = Y.reshape(ell, kx * ky, 3, 1)
            else:
                Y = Y.reshape(ell, kx, ky, 3, 1)

            if i == 0:
                X = Y
            elif self.unfold_space or self.unfold_all:
                X = np.append(X, Y, axis=3)  # X is class ndarray
            else:
                X = np.append(X, Y, axis=4)  # X is class ndarray
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

    def display_dictionary_CP(self, W, plot_shape_N_color=False):
        k = self.patch_size
        num_rows = np.ceil(np.sqrt(self.n_components)).astype(int)
        num_cols = np.ceil(np.sqrt(self.n_components)).astype(int)

        U0 = W.get('U0') ### dict for time mode
        U1 = W.get('U1') ### dict for shape & color mode
        exemption_list = [15, 16, 17, 18, 19]
        idx = [i for i in np.arange(25).tolist() if not i in exemption_list]
        U0 = U0[:, idx]
        U1 = U1[:, idx]

        # cv2.imshow('frame 10', U1[:,5].reshape(self.frameHeight, self.frameWidth, 3))
        ### shape mode
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6, 6),
                                 subplot_kw={'xticks': [], 'yticks': []})
        fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(6, 4.2),
                                 subplot_kw={'xticks': [], 'yticks': []})
        for ax, i in zip(axs.flat, range(self.n_components)):
            patch = U1[:,i].reshape(self.frameHeight, self.frameWidth, 3)
            ### for color inversion
            ax.imshow(0.8 - patch/np.max(patch))

        plt.tight_layout()
        plt.suptitle('Shape mode', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

        ### color mode
        # fig, axs = plt.subplots(nrows=self.n_components, ncols=1, figsize=(6, 6),
        #                        subplot_kw={'xticks': [], 'yticks': []})
        fig, axs = plt.subplots(nrows=self.n_components-5, ncols=1, figsize=(6, 4.2),
                                subplot_kw={'xticks': [], 'yticks': []})
        for ax, i in zip(axs.flat, range(self.n_components)):
            ax.imshow(U0[:, i].reshape(1, -1), cmap="gray", interpolation='nearest')

        plt.tight_layout()
        plt.suptitle('Time mode', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

        '''
        ### Combined CPdict
        CPdict = self.out(W)
        #fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6, 6),
        #                        subplot_kw={'xticks': [], 'yticks': []})
        fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(3, 6),
                                subplot_kw={'xticks': [], 'yticks': []})

        for ax, i in zip(axs.flat, range(self.n_components)):
            patch = CPdict.get('A'+str(i)).reshape(k, k, 3)
            ax.imshow(patch / np.max(patch))

        plt.tight_layout()
        plt.suptitle('Dictionary learned from \n patches of size %d' % k, fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()
        '''

    def display_dictionary_CP_combined(self, W, save_folder, grid=None):
        k = self.patch_size

        U0 = W.get('U0') ### dict for time mode
        U1 = W.get('U1') ### dict for shape mode

        print('U0.shape', U0.shape)
        print('U0_0', U0[:,0])
        print('U1.shape', U1.shape)


        ### Use the ordering of spatial modes from the brightest to the darkest
        importance = np.sum(np.abs(U1), axis=0)  # l1 norms of colums of U1
        idx = np.argsort(importance)
        #idx = np.flip(idx)

        ### Plot separately the first atom

        for i in np.arange(U1.shape[1]):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6),
                                     subplot_kw={'xticks': [], 'yticks': []})
            patch = U1[:,idx[i]].reshape(self.frameHeight, self.frameWidth, 3)
            ax.imshow((np.max(patch) - patch)/np.max(patch))
            # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
            plt.savefig(save_folder + '/fig_topics_new_atom'+str(i))



        ### Make gridspec
        fig1 = plt.figure(figsize=(13.1, 4), constrained_layout=False)
        # make outer gridspec
        outer_rows = 1
        outer_cols = 3
        outer_grid = gridspec.GridSpec(nrows=outer_rows, ncols=outer_cols, wspace=0.02, hspace=0.05, width_ratios=[4.1, 5, 2])

        if grid is not None:
            num_atoms2show = grid[0]*grid[1]
        else:
            num_atoms2show = U1.shape[1]


        ### Set up number of rows and colums for spatial atoms
        num_rows = np.floor(np.sqrt(num_atoms2show)).astype(int)
        if num_rows ** 2 == num_atoms2show:
            num_cols = num_rows
        else:
            num_cols = num_rows + 1
        if grid is not None:
            [num_rows, num_cols] = grid


        # U0 = U0[:,idx[:num_atoms2show]]
        # U1 = U1[:,idx[:num_atoms2show]]
        print('!!! U0.shape', U0.shape)
        print('!!! U1.shape', U1.shape)

        Ndict_wspace = 0.05
        Ndict_hspace = 0.05

        # make nested gridspecs
        for i in range(outer_rows * outer_cols):
            if i % outer_cols == 0:
                atom_num = 9
                inner_grid = outer_grid[i].subgridspec(1, 1, wspace=Ndict_wspace, hspace=Ndict_hspace)
                ax = fig1.add_subplot(inner_grid[0, 0])
                patch = U1[:,idx[atom_num-1]].reshape(self.frameHeight, self.frameWidth, 3)
                ax.imshow((np.max(patch) - patch)/np.max(patch))
                #ax.title.set_text('Spatial Activation Atom # '+str(atom_num))

            elif i % outer_cols == 1:

                # display grid of spatial atoms
                print('!!!! num_rows', num_rows)
                print('!!!! num_cols', num_cols)


                inner_grid = outer_grid[i].subgridspec(num_rows, num_cols, wspace=Ndict_wspace, hspace=Ndict_hspace)
                for j in np.arange(num_atoms2show):
                    a = j // num_cols
                    b = j % num_cols

                    ax = fig1.add_subplot(inner_grid[a, b])
                    patch = U1[:,idx[j]].reshape(self.frameHeight, self.frameWidth, 3)
                    ### for color inversion
                    ax.imshow((np.max(patch) - patch)/np.max(patch))
                    #ax.set_xlabel('%1.2f' % importance[idx[j]], fontsize=10)  # get the largest first
                    #ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    ax.set_xticks([])
                    ax.set_yticks([])
                    #if a == 0 and b == 2:
                    #    ax.title.set_text('Spatial Activation')

            else:
                # display temporal prevalence
                inner_grid = outer_grid[i].subgridspec(1, num_atoms2show+1, wspace=0.5, hspace=0.05) # +1 for colorbar
                for j in np.arange(num_atoms2show):
                    ax = fig1.add_subplot(inner_grid[0, j])
                    b = U0[:, idx[j]].reshape(1, -1).T
                    im2 = ax.imshow(b, cmap="viridis",  ## black = 1, white = 0
                              interpolation='nearest', aspect='auto', vmin=0, vmax=np.max(U0))
                    ax.set_xticks([])
                    j1 = j+1
                    ax.set_xlabel('%i' % j1, fontsize=9)  # get the largest first
                    # ax.xaxis.set_label_coords(0.5, 0.05)  # adjust location of importance appearing beneath patches
                    if j1 % 2 == 1:
                        ax.xaxis.set_label_position('top')
                    else:
                        ax.xaxis.set_label_position('bottom')

                    if j<num_atoms2show-1:
                        ax.set_yticks([])
                    else:
                        ax.set_ylabel('time')
                        # yticks = np.arange(0,U0.shape[0],3)*0.04
                        # yticks = [str(value) for value in yticks]
                        # ax.set_yticks(yticks)
                        ax.yaxis.tick_right()
                        ax.yaxis.set_label_position("right")


                    #if j == num_atoms2show // 2:
                    #    ax.title.set_text('Temporal Activation')

                    #if j==U0.shape[1]-1:
                    #    divider = make_axes_locatable(ax)
                    #    cax = divider.append_axes('right', size='50%', pad=0.05)
                    #    fig1.colorbar(im2, cax=cax, orientation='vertical')


        # plt.tight_layout()
        # plt.suptitle('Shape mode', fontsize=16)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.01, hspace=0.01)
        #plt.figtext(0.2, 1, 'Spatial activation pattern # ' + str(i), fontsize=12)
        #plt.figtext(0.55, 1, "Spatial Activation", ha="center", va="top", fontsize=12)
        #plt.figtext(0.9, 1, "Temporal Activation", ha="center", va="top", fontsize=12)


        plt.savefig(save_folder + '/fig_combined_modes')


    def display_dictionary(self, W, learn_joint_dict=False, color_mode=False, display_out_dict=True):
        k = self.patch_size
        num_rows = np.ceil(np.sqrt(self.n_components)).astype(int)
        num_cols = np.ceil(np.sqrt(self.n_components)).astype(int)

        if color_mode:
            fig, axs = plt.subplots(nrows=1, ncols=self.n_components, figsize=(6, 2),
                                    subplot_kw={'xticks': [], 'yticks': []})
        else:
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6, 6),
                                    subplot_kw={'xticks': [], 'yticks': []})

        for ax, i in zip(axs.flat, range(self.n_components)):
            if color_mode:
                ax.imshow(W[:, i].reshape(-1, 1))
            elif not learn_joint_dict:
                ax.imshow(W[:, i].reshape(k, k), cmap="gray", interpolation='nearest')
            else:
                patch = W[:, i].reshape(k, k, 3)
                ax.imshow(patch / np.max(patch))

        plt.tight_layout()
        plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

    def get_downscaled_dims(self, path, downscale_factor=None, is_matrix=False):
        # need to put is_matrix arg at the end to avoid error (don't know why)
        if downscale_factor is None:
            downscale_factor = self.downscale_factor

        if not is_matrix:
            img = Image.open(path)
            img = np.asarray(img.convert('L'))
        else:
            img = np.load(path)

        data = downscale_local_mean(img, (downscale_factor, downscale_factor))
        return data.shape[0], data.shape[1]

    def train_dict(self):
        print('training CP dictionaries from patches...')
        '''
        Trains dictionary based on patches from an i.i.d. sequence of batch of patches
        CP dictionary learning
        '''
        W = self.W
        At = []
        Bt = []
        code = self.code
        for t in np.arange(self.iterations):
            X = self.extract_random_patches()
            print('X.shape', X.shape)
            # plt.imshow(X.reshape(self.frameHeight, self.frameWidth, 3))
            if t == 0:
                self.ntf = Online_CPDL(X,
                                       self.n_components,
                                       iterations=self.sub_iterations,
                                       batch_size=self.batch_size,
                                       subsample=False)
                W, At, Bt, H = self.ntf.train_dict()
            else:
                self.ntf = Online_CPDL(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_loading=W,
                                      ini_A=At,
                                      ini_B=Bt,
                                      history=self.ntf.history,
                                      subsample=False)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                W, At, Bt, H = self.ntf.train_dict()
                # code += H
            print('Current iteration %i out of %i' % (t, self.iterations))
        self.W = self.ntf.loading
        # print('dict_shape:', self.W.shape)
        # print('code_shape:', self.code.shape)
        np.save('Video_dictionary/dict_learned_CPDL' , self.W)
        np.save('Video_dictionary/code_learned_CPDL' , self.code)
        return W

    def show_array(self, arr):
        plt.figure()
        plt.imshow(arr, cmap="gray")
        plt.show()

    def reconstruct_image_color(self, loading, recons_resolution=1, if_save=True):
        print('reconstructing given network...')
        '''
        Reconstruct original color image using lerned CP dictionary atoms
        '''
        A = self.data  # A.shape = (row, col, 3)
        CPdict = self.out(loading)
        k = self.patch_size
        W = np.zeros(shape=(3*k**2, self.n_components))
        for j in np.arange(self.n_components):
            W[:, j] = CPdict.get('A' + str(j)).reshape(-1, 1)[:, 0]

        A_matrix = A.reshape(-1, A.shape[1])  # (row, col, 3) --> (3row, col)
        [m, n] = A_matrix.shape
        A_recons = np.zeros(shape=A.shape)
        A_overlap_count = np.zeros(shape=(A.shape[0], A.shape[1]))
        k = self.patch_size
        t0 = time()
        c = 0
        num_rows = np.floor((A_recons.shape[0]-k)/recons_resolution).astype(int)
        num_cols = np.floor((A_recons.shape[1]-k)/recons_resolution).astype(int)

        for i in np.arange(0, A_recons.shape[0]-k, recons_resolution):
            for j in np.arange(0, A_recons.shape[1]-k, recons_resolution):
                patch = A[i:i + k, j:j + k, :]
                patch = patch.reshape((-1, 1))
                coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                    transform_alpha=1, transform_algorithm='lasso_lars', positive_code=True)
                # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
                code = coder.transform(patch.T)
                patch_recons = np.dot(W, code.T).T
                patch_recons = patch_recons.reshape(k, k, 3)

                # now paint the reconstruction canvas
                for x in itertools.product(np.arange(k), repeat=2):
                    c = A_overlap_count[i+x[0], j+x[1]]
                    A_recons[i+x[0], j+x[1], :] = (c*A_recons[i+x[0], j+x[1], :] + patch_recons[x[0], x[1], :])/(c+1)
                    A_overlap_count[i+x[0], j+x[1]] += 1

                # progress status
                print('reconstructing (%i, %i)th patch out of (%i, %i)' % (i/recons_resolution, j/recons_resolution, num_rows, num_cols))
        print('Reconstructed in %.2f seconds' % (time() - t0))
        print('A_recons.shape', A_recons.shape)
        if if_save:
            np.save('Video_dictionary/video_recons_color', A_recons)
        plt.imshow(A_recons)
        return A_recons
