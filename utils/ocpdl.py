# (setq python-shell-interpreter "./venv/bin/python")

import numpy as np
import progressbar
# import imageio
import matplotlib.pyplot as plt
from numpy import linalg as LA
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly import unfold as tl_unfold
from tensorly.decomposition import parafac
from sklearn.decomposition import SparseCoder

DEBUG = False


class Online_CPDL():

    def __init__(self,
                 X, n_components=100,
                 iterations=500,
                 sub_iterations=10,
                 batch_size=20,
                 ini_CPdict=None,
                 ini_loading=None,
                 history=0,
                 ini_A = None,
                 ini_B = None,
                 alpha=None,
                 beta=None,
                 subsample=True):
        '''
        Online CP Dictionary Learning algorithm
        X: data tensor (n+1 -dimensional) with shape I_1 * I_2 * ... * I_n * I_n+1
        Last node considered as the "batch mode"
        Seeks to find CP dictionary D = [A1, A2, ... , A_R], R=n_components, Ai = rank 1 tensor
        Such that each slice X[:,:,..,:, j] \approx <D, c> for some c

        n_components (int) = r = number of rank-1 CP factors
        iter (int): number of iterations where each iteration is a call to step(...)
        batch_size (int): number random of columns of X that will be sampled during each iteration
        '''
        self.X = X
        self.n_modes = X.ndim -1 ### -1 for discounting the last "batch_size" mode
        self.n_components = n_components
        self.batch_size = batch_size
        self.iterations = iterations
        self.sub_iterations = sub_iterations

        if ini_CPdict is not None:
            self.CPdict = ini_CPdict
        else:
            self.CPdict  = self.initialize_CPdict()

        if ini_loading is None:
            self.loading = self.initialize_loading()
        else:
            self.loading = ini_loading

        if ini_A is not None:
            self.ini_A = ini_A
        else:
            self.ini_A = np.zeros(shape=(n_components, n_components))
        # print('A', self.ini_A)

        if ini_A is not None:
            self.ini_B = ini_B
        else:
            Y = X.take(indices=0, axis=-1)
            self.ini_B = np.zeros(shape=(len(Y.reshape(-1,1)), n_components))

        self.history = history
        self.alpha = alpha
        self.beta = beta
        self.code = np.zeros(shape=(n_components, X.shape[-1]))  ### X.shape[-1] = batch_size
        self.subsample = subsample

    def initialize_loading(self):
        ### loading = python dict of [U1, U2, \cdots, Un], each Ui is I_i x R array
        loading = {}
        for i in np.arange(self.n_modes):
            loading.update({'U'+str(i): np.random.rand(self.X.shape[i], self.n_components) })
        return loading

    def initialize_CPdict(self):
        ### CPdict = python dict of [A1, A2, \cdots, AR], R=n_components, each Ai is a rank-1 tensor
        CPdict = {}
        for i in np.arange(self.n_components):
            CPdict.update({'A' + str(i): np.zeros(shape=self.X.shape[:-1])})
            ### Exclude last "batch size" mode
        return CPdict

    def out(self, loading):
        ### given loading, take outer product of respected columns to get CPdict
        CPdict = {}
        for i in np.arange(self.n_components):
            A = np.array([1])
            for j in np.arange(self.n_modes):
                loading_factor = loading.get('U'+str(j))  ### I_i X n_components matrix
                # print('loading_factor', loading_factor)
                A = np.multiply.outer(A, loading_factor[:,i])
            A = A[0]
            CPdict.update({'A' + str(i): A})
        return CPdict

    def sparse_code_tensor(self, X, CPdict):
        '''
        Given data tensor X and CP dictionary CPdict, find sparse code c such that
        X \approx <CPdict, c>

        args:
            X (numpy array): data tensor with dimensions: (I1) x (I2) x ... x (In) x (In+1)
            CPdict (numpy dictionary): [A1, ... AR], R=n_componetns, Ai = rank 1 tensor

        returns:
            c (numpy array): sparse code with dimensions: topics (n_components) x samples (In+1)

        method:
            matricise X into (I1 ... In) x (In+1) and Ai's and use the usual sparse coding
        '''

        ### make a dictionary matrix from CPdict
        W = np.zeros(shape=(len(X.reshape(-1,X.shape[-1])), self.n_components))
        for j in np.arange(self.n_components):
            W[:, j] = CPdict.get('A'+str(j)).reshape(-1,1)[:,0]

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', CPdict.shape, '\n')

        # initialize the SparseCoder with W as its dictionary
        # then find H such that X \approx W*H
        if self.alpha == None:
            coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=2, transform_algorithm='lasso_lars', positive_code=True)
        else:
            coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=1, transform_algorithm='lasso_lars', positive_code=True)
        # alpha = L1 regularization parameter.
        c = coder.transform(X.reshape(-1,X.shape[-1]).T)

        # transpose H before returning to undo the preceding transpose on X
        return c

    def get_A_U(self, A, loading, j):
        ### Compute (n_componetns) x (n_components) intermediate aggregation matrix
        A_U = A
        for k in np.arange(self.n_modes):
            if k != j:
                U = loading.get('U'+str(k))  ### I_{k} x n_components loading matrix
                U = U.T @ U ### (n_componetns) x (n_components) matrix
                A_U = A_U * U
        return A_U

    def get_B_U(self, B, loading, j):
        ### Compute (n_components) x (I_j) intermediate aggregation matrix
        ### B has size (I_1 * I_2 * ... * I_n) x (n_componetns)

        B_U = np.zeros(shape=(self.n_components, self.X.shape[j]))
        for r in np.arange(self.n_components):
            B_r = B[:,r].reshape(self.X.shape[:-1])
            for i in np.arange(self.n_modes):
                if i != j:
                    U = loading.get('U'+str(i))  ### I_{k} x n_components loading matrix
                    B_r = tl.tenalg.mode_dot(B_r, U[:,r], i)
                    B_r = np.expand_dims(B_r, axis=i)  ## mode_i product above kills axis i
            B_U[r, :] = B_r.reshape(self.X.shape[j])
        return B_U

    def update_dict(self, loading, A, B):
        '''
        Updates loading = [U1, .. , Un] using new aggregate matrices A and B

        args:
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (R)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(R)
            B (numpy array): aggregate matrix with dimensions: (I1 ... In) x topics (R)

          returns:
            loading = [U1', .. , Un'] (numpy dict): each Ui has shape I_i x R
        '''
        # extract matrix dimensions from W
        # and initializes the copy W1 that is updated in subsequent for loop

        for j in np.arange(self.n_modes):
            A_U = self.get_A_U(A, loading, j)  ### intermediate aggregate matrices updated for each j
            B_U = self.get_B_U(B, loading, j)  ### intermediate aggregate matrices updated for each j
            W1 = loading.get('U'+str(j)).copy()
            for k in np.arange(self.n_components):
                # W1[:,k] = W1[:,k] - (1/W1[k, k])*(np.dot(W1, A[:,k]) - B.T[:,k])
                W1[:, k] = W1[:, k] - (1 / (A_U[k, k] + 1)) * (np.dot(W1, A_U[:, k]) - B_U[k,:])
                W1[:, k] = np.maximum(W1[:, k], np.zeros(shape=(W1.shape[0],)))
                W1[:, k] = (1 / np.maximum(1, LA.norm(W1[:, k]))) * W1[:, k]
            loading.update({'U'+str(j):W1}) ### Lindeberg replacement trick here
        return loading

    def step(self, X, A, B, loading, t):
        '''
        Performs a single iteration of the online NMF algorithm from
        Han's Markov paper.
        Note: H (numpy array): code matrix with dimensions: topics (r) x samples(n)

        args:
            X (numpy array): data matrix with dimensions: data_dim (d) x samples (n)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(r)
            B (numpy array): aggregate matrix with dimensions: topics (r) x data_dim(d)
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
            t (int): current iteration of the online algorithm

        returns:
            Updated versions of H, A, B, and W after one iteration of the online NMF
            algorithm (H1, A1, B1, and W1 respectively)
        '''

        # Compute H1 by sparse coding X using dictionary W
        CPdict = self.out(loading)
        H1 = self.sparse_code_tensor(X, CPdict)

        if DEBUG:
            print(H1.shape)

        # Update aggregate matrices A and B
        t = t.astype(float)
        if self.beta == None:
            beta = 1
        else:
            beta = self.beta
        A1 = (1 - (t ** (-beta))) * A + t ** (-beta) * np.dot(H1.T, H1)
        B1 = (1 - (t ** (-beta))) * B + t ** (-beta) * np.dot(X.reshape(-1, X.shape[-1]), H1)

        # Update dictionary matrix
        loading1 = self.update_dict(loading, A1, B1)
        self.history = t + 1
        # print('history=', self.history)
        return H1, A1, B1, loading1

    def train_dict(self):
        '''
        Given data tensor X, learn loading matrices L=[U0, U1, \cdots, Un-1] that gives CPdict = out(L).
        '''

        A = self.ini_A
        B = self.ini_B
        loading = self.loading
        code = self.code
        t0 = self.history

        for i in np.arange(1, self.iterations):
            X_batch = self.X
            idx = np.arange(self.X.shape[-1])
            # randomly choose batch_size number of "columns" to sample
            # initializing the "batch" of X, which are the subset
            # of columns from X_unfold that were randomly chosen above
            if self.subsample:
                idx = np.random.randint(self.X.shape[-1], size=self.batch_size)
                X_batch = X_batch.take(indices=idx, axis=-1)

            # iteratively update "loading" using batches of X, along with
            # iteratively updating values of A and B
            # print('X.shape before training step', self.X.shape)
            H, A, B, loading = self.step(X_batch, A, B, loading, t0 + i)
            code[:, idx] += H.T
            # print('dictionary=', W)
            # print('code=', H)
            # plt.matshow(H)
            error = self.compute_reconstruction_error(X_batch, loading)
            print('error', error)

            #  progress status
            # print('Current iteration %i out of %i' % (i, self.iterations))
        self.CPdict = self.out(loading)  ### Rank 1 CP factors
        self.loading = loading
        self.code = code
        return loading, A, B, code

    def inner_product_loading_code(self, loading, code):
        ### loading = [U1, .. , Un], Ui has shape Ii x R
        ### code = shape R x batch_size
        ### compute the reconstructed tensor of size I1 x I2 x .. x In x batch_size
        ### from the given loading = [U1, .. , Un] and code = [c1, .., c_batch_size], ci = (R x 1)
        recons = np.zeros(shape=self.X.shape[:-1])
        recons = np.expand_dims(recons, axis=-1)  ### Now shape I1 x I2 x .. x In x 1
        CPdict = self.out(loading)
        for i in np.arange(code.shape[1]):
            A = np.zeros(self.X.shape[:-1])
            for j in np.arange(len(loading.keys())):
                A = A + CPdict.get('A'+str(j))*code[j,i]
            A = np.expand_dims(A, axis=-1)
            recons = np.concatenate((recons, A), axis=-1)
        recons = np.delete(recons, obj=0 ,axis=-1)
        # print('recons.shape', recons.shape)
        return recons

    def compute_reconstruction_error(self, X, loading):
        ### X data tensor, loading = loading matrices
        ### Find sparse code and compute reconstruction error
        c = self.sparse_code_tensor(X, self.out(loading))
        # print('X.shape', X.shape)
        # print('c.shape', c.shape)
        recons = self.inner_product_loading_code(loading, c.T)
        error = np.linalg.norm((X-recons).reshape(-1,1), ord=2)
        return error