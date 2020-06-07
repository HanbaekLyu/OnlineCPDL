from utils.ocpdl import Online_CPDL
import numpy as np
from PIL import Image
from skimage.transform import downscale_local_mean
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition import SparseCoder
from time import time
import itertools
import matplotlib.pyplot as plt

DEBUG = False


class Image_Reconstructor_OCPDL():
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
                 patches_file='',
                 alpha=1,
                 learn_joint_dict=False,
                 is_matrix=False,
                 unfold_space=False,
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
        self.is_color = is_color
        self.alpha = alpha ## sparsity regularizer
        self.W = np.zeros(shape=(patch_size, n_components))
        self.code = np.zeros(shape=(n_components, iterations*batch_size))

        # read in image as array
        self.data = self.read_img_as_array()

    def read_img_as_array(self):
        '''
        Read input image as a narray
        '''

        if self.is_matrix:
            img = np.load(self.path)
            data = (img + 1) / 2  # it was +-1 matrix; now it is 0-1 matrix
        else:
            img = Image.open(self.path)
            if self.is_color:
                img = img.convert('RGB')
            else:
                img = img.convert('L')
            # normalize pixel values (range 0-1)
            data = np.asarray(img) / 255
        print('data.shape', data.shape)
        return data

    def extract_random_patches(self):
        '''
        Extract 'num_patches' many random patches of given size
        Three tensor data types depending on how to unfold k by k by 3 color patches:
            unfold_space : k**2 by 3
            unfold_all : 3*k**2 by 1
            else: k by k by 3
        '''
        x = self.data.shape
        k = self.patch_size
        num_patches = self.num_patches

        if self.unfold_all:
            X = np.zeros(shape=(3 * (k ** 2), 1, 1))
        elif self.unfold_space:
            X = np.zeros(shape=(k ** 2, 3, 1))
        else:
            X = np.zeros(shape=(k, k, 3, 1))

        for i in np.arange(num_patches):
            a = np.random.choice(x[0] - k)  # x coordinate of the top left corner of the random patch
            b = np.random.choice(x[1] - k)  # y coordinate of the top left corner of the random patch
            Y = self.data[a:a + k, b:b + k, :] # k by k by 3

            if self.unfold_all:
                Y = Y.reshape(3 * (k ** 2), 1, 1)

            elif self.unfold_space:
                Y = Y.reshape(k ** 2, 3, 1)
            else:
                Y = Y.reshape(k, k, 3, 1)

            if i == 0:
                X = Y
            elif self.unfold_space or self.unfold_all:
                X = np.append(X, Y, axis=2)  # x is class ndarray
            else:
                X = np.append(X, Y, axis=3)  # x is class ndarray
        return X

    def image_to_patches(self, path, patch_size=10, downscale_factor=2, is_matrix=False, is_recons=False):
        '''
        #*****
        args:
            path (string): Path and filename of input image
            patch_size (int): Pixel dimension of square patches taken of image
            color (boolean): Specifies conversion of image to RGB (True) or grayscale (False).
                Default value = false. When color = True, images in gray colorspace will still appear
                gray, but will thereafter be represented in RGB colorspace using three channels.
            downscale_factor: Specifies the extent to which the image will be downscaled. Greater values
                will result in more downscaling but faster speed. For no downscaling, use downscale_factor=1.
        returns: #***
        '''
        #open image and convert it to either RGB (three channel) or grayscale (one channel)
        if is_matrix:
            img = np.load(path)
            data = (img + 1) / 2  # it was +-1 matrix; now it is 0-1 matrix
        else:
            img = Image.open(path)
            if self.is_color:
                img = img.convert('RGB')
            else:
                img = img.convert('L')
            # normalize pixel values (range 0-1)
            data = np.asarray(img) / 255

        if DEBUG:
            print(np.asarray(img))

        patches = self.extract_random_patches()
        print('patches.shape=', patches.shape)
        return patches

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

        if plot_shape_N_color:
            U0 = W.get('U0') ### dict for shape mode
            U1 = W.get('U1') ### dict for color mode

            ### shape mode
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6, 6),
                                    subplot_kw={'xticks': [], 'yticks': []})
            for ax, i in zip(axs.flat, range(self.n_components)):
                ax.imshow(U0[:,i].reshape(k, k), cmap="gray", interpolation='nearest')

            plt.tight_layout()
            plt.suptitle('Learned dictionary for shape mode  \n from patches of size %d' % k, fontsize=16)
            plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
            plt.show()

            ### color mode
            fig, axs = plt.subplots(nrows=self.n_components, ncols=1, figsize=(2, 6),
                                    subplot_kw={'xticks': [], 'yticks': []})
            for ax, i in zip(axs.flat, range(self.n_components)):
                ax.imshow(U1[:, i].reshape(1, -1), cmap="gray", interpolation='nearest')

            plt.tight_layout()
            plt.suptitle('Color mode', fontsize=16)
            plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
            plt.show()

        ### Combined CPdict
        CPdict = self.out(W)
        print('CPdict', CPdict)
        #fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6, 6),
        #                         subplot_kw={'xticks': [], 'yticks': []})
        fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(3, 6),
                                subplot_kw={'xticks': [], 'yticks': []})

        for ax, i in zip(axs.flat, range(self.n_components)):
            patch = CPdict.get('A'+str(i)).reshape(k, k, 3)
            print('patch.color', patch[:,:,])
            ax.imshow(patch / np.max(patch))

        plt.tight_layout()
        # plt.suptitle('Dictionary learned from \n patches of size %d' % k, fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

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
            if t == 0:
                self.ntf = Online_CPDL(X,
                                       self.n_components,
                                       iterations=self.sub_iterations,
                                       batch_size=self.batch_size,
                                       alpha=self.alpha,
                                       subsample=False)
                W, At, Bt, H = self.ntf.train_dict()
            else:
                self.ntf = Online_CPDL(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_loading=W,
                                      ini_A=At,
                                      ini_B=Bt,
                                      alpha=self.alpha,
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
        np.save('Image_dictionary/dict_learned_CPDL' , self.W)
        np.save('Image_dictionary/code_learned_CPDL' , self.code)
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
            np.save('Image_dictionary/img_recons_color', A_recons)
        plt.imshow(A_recons)
        return A_recons







