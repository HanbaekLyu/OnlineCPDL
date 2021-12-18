# (setq python-shell-interpreter "./venv/bin/python")

import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from sklearn.decomposition import SparseCoder
import time
from tqdm import trange

DEBUG = False


class ALS_DR():

    def __init__(self,
                 X, n_components=100,
                 iterations=500,
                 sub_iterations=10,
                 batch_size=20,
                 ini_CPdict=None,
                 ini_loading=None,
                 history=0,
                 ini_A=None,
                 ini_B=None,
                 alpha=None,
                 beta=None,
                 subsample=True):
        '''
        Alternating Least Squares with Diminishing Radius for Nonnegative CP Tensor Factorization
        X: data tensor (n-dimensional) with shape I_1 * I_2 * ... * I_n
        Seeks to find n loading matrices U0, ... , Un, each Ui is (I_i x r)
        n_components (int) = r = number of rank-1 CP factors
        iter (int): number of iterations where each iteration is a call to step(...)
        '''
        self.X = X
        self.n_modes = X.ndim - 1  ### -1 for discounting the last "batch_size" mode
        self.n_components = n_components
        self.batch_size = batch_size
        self.iterations = iterations

        if ini_CPdict is not None:
            self.CPdict = ini_CPdict
        else:
            self.CPdict = self.initialize_CPdict()

        if ini_loading is None:
            self.loading = self.initialize_loading()
        else:
            self.loading = ini_loading

        self.alpha = alpha
        self.X_norm = np.linalg.norm(X.reshape(-1, 1), ord=2)

    def initialize_loading(self):
        ### loading = python dict of [U1, U2, \cdots, Un], each Ui is I_i x R array
        loading = {}
        for i in np.arange(self.n_modes):
            loading.update({'U' + str(i): np.random.rand(self.X.shape[i], self.n_components)})
        return loading

    def initialize_CPdict(self):
        ### CPdict = python dict of [A1, A2, \cdots, AR], R=n_components, each Ai is a rank-1 tensor
        CPdict = {}
        for i in np.arange(self.n_components):
            CPdict.update({'A' + str(i): np.zeros(shape=self.X.shape[:-1])})
            ### Exclude last "batch size" mode
        return CPdict

    def out(self, loading, drop_last_mode=False):
        ### given loading, take outer product of respected columns to get CPdict
        ### Use drop_last_mode for ALS
        CPdict = {}
        for i in np.arange(self.n_components):
            A = np.array([1])
            if drop_last_mode:
                n_modes_multiplied = len(loading.keys()) - 1
            else:
                n_modes_multiplied = len(loading.keys())  # also equals self.X_dim - 1
            for j in np.arange(n_modes_multiplied):
                loading_factor = loading.get('U' + str(j))  ### I_i X n_components matrix
                # print('loading_factor', loading_factor)
                A = np.multiply.outer(A, loading_factor[:, i])
            A = A[0]
            CPdict.update({'A' + str(i): A})
        return CPdict

    def update_code_within_radius(self, X, W, H0, r, alpha=0,
                                  sub_iter=[2],
                                  subsample_ratio=None, nonnegativity=True,
                                  use_line_search=False):
        '''
        Find \hat{H} = argmin_H ( | X - WH| + alpha|H| ) within radius r from H0
        Use row-wise projected gradient descent
        Do NOT sparsecode the whole thing and then project -- instable
        12/5/2020 Lyu

        For NTF problems, X is usually tall and thin so it is better to subsample from rows
        12/25/2020 Lyu

        Apply single round of AdaGrad for rows, stop when gradient norm is small and do not make update
        12/27/2020 Lyu
        '''

        # print('!!!! X.shape', X.shape)
        # print('!!!! W.shape', W.shape)
        # print('!!!! H0.shape', H0.shape)

        H1 = H0.copy()
        i = 0
        dist = 1
        idx = np.arange(X.shape[0])
        H1_old = H1.copy()

        A = W.T @ W
        B = W.T @ X

        while (i < np.random.choice(sub_iter)):
            #if_continue = np.ones(H0.shape[0])  # indexed by rows of H

            for k in [k for k in np.arange(H0.shape[0])]:

                grad = (np.dot(A[k, :], H1) - B[k, :] + alpha * np.sign(H1[k, :]) * np.ones(H0.shape[1]))
                grad_norm = np.linalg.norm(grad, 2)

                # Initial step size
                step_size = 1/(A[k,k]+1)
                # step_size = 1 / (np.trace(A)) # use the whole trace
                # step_size = 1
                if r is not None:  # usual sparse coding without radius restriction
                    d = step_size * grad_norm
                    step_size = (r / max(r, d)) * step_size

                H1_temp = H1.copy()
                # loss_old = np.linalg.norm(X - W @ H1)**2
                H1_temp[k, :] = H1[k, :] - step_size * grad
                if nonnegativity:
                    H1_temp[k,:] = np.maximum(H1_temp[k,:], np.zeros(shape=(H1.shape[1],)))  # nonnegativity constraint
                #loss_new = np.linalg.norm(X - W @ H1_temp)**2
                #if loss_old > loss_new:

                    # print('recons_loss:' , np.linalg.norm(X - W @ H1, ord=2) / np.linalg.norm(X, ord=2))

                if use_line_search:
                # Armijo backtraking line search
                    m = grad.T @ H1[k,:]
                    H1_temp = H1.copy()
                    loss_old = np.linalg.norm(X - W @ H1)**2
                    loss_new = 0
                    count = 0
                    while (count==0) or (loss_old - loss_new < 0.1 * step_size * m):
                        step_size /= 2
                        H1_temp[k, :] = H1[k, :] - step_size * grad
                        if nonnegativity:
                            H1_temp[k,:] = np.maximum(H1_temp[k,:], np.zeros(shape=(H1.shape[1],)))  # nonnegativity constraint
                        loss_new = np.linalg.norm(X - W @ H1_temp)**2
                        count += 1

                H1 = H1_temp

            i = i + 1


        return H1

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
                A = A + CPdict.get('A' + str(j)) * code[j, i]
            A = np.expand_dims(A, axis=-1)
            recons = np.concatenate((recons, A), axis=-1)
        recons = np.delete(recons, obj=0, axis=-1)
        # print('recons.shape', recons.shape)
        return recons

    def compute_reconstruction_error(self, X, loading, is_X_full_tensor=False):
        ### X data tensor, loading = loading matrices
        ### Find sparse code and compute reconstruction error
        ### If X is full tensor,
        # c = self.sparse_code_tensor(X, self.out(loading))
        c = np.ones(len(loading.keys()))
        # print('X.shape', X.shape)
        # print('c.shape', c.shape)
        recons = self.inner_product_loading_code(loading, c.T)
        error = np.linalg.norm((X - recons).reshape(-1, 1), ord=2)
        return error

    def ALS(self,
            iter=100,
            ini_loading=None,
            beta = None,
            if_compute_recons_error=False,
            save_folder='Output_files',
            search_radius_const = 1000,
            subsample_ratio=None,
            nonnegativity=True,
            output_results=False):
        '''
        Given data tensor X and initial loading matrices W_ini, find loading matrices W by Alternating Least Squares
        with Diminishing Radius
        '''

        X = self.X
        normalization = np.linalg.norm(X.reshape(-1,1),1)/np.product(X.shape) # avg entry of X
        print('!!! avg entry of X', normalization)
        #X = X/normalization

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', ini_loading.shape, '\n')

        n_modes = len(X.shape)
        if ini_loading is not None:
            loading = ini_loading.copy() # put .copy() Otherwise initial loading would change
        else:
            loading = {}
            for i in np.arange(X.ndim):
                loading.update({'U' + str(i): np.random.rand(X.shape[i], self.n_components)})

        result_dict = {}
        time_error = np.zeros(shape=[0, 2])
        elapsed_time = 0


        # Initial reconstruction error
        if if_compute_recons_error:
            error = self.compute_recons_error(data=X, loading=loading)
            # error *= normalization
            time_error = np.append(time_error, np.array([[0, error]]), axis=0)
            print('!!! Reconstruction error at iteration %i = %f.3' % (0, error))

        for step in trange(int(iter)):
            start = time.time()

            for mode in np.arange(n_modes):
                X_new = np.swapaxes(X, mode, -1)

                U = loading.get('U' + str(mode))  # loading matrix to be updated
                loading_new = loading.copy()
                loading_new.update({'U' + str(mode): loading.get('U' + str(n_modes - 1))})
                loading_new.update({'U' + str(n_modes - 1): U})

                X_new_mat = X_new.reshape(-1, X_new.shape[-1])
                CPdict = self.out(loading_new, drop_last_mode=True)
                W = np.zeros(shape=(X_new_mat.shape[0], self.n_components))
                for j in np.arange(self.n_components):
                    W[:, j] = CPdict.get('A' + str(j)).reshape(-1, 1)[:, 0]

                if beta is None:  # usual nonnegative sparse coding
                    """
                    coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                        transform_alpha=0, transform_algorithm='lasso_lars', positive_code=True)
                    # alpha = L1 regularization parameter.
                    Code = coder.transform(X_new_mat.T)
                    """
                    Code = self.update_code_within_radius(X_new_mat, W, U.T, r=None, alpha=0,
                                                        subsample_ratio=1,
                                                        sub_iter = [30],
                                                        nonnegativity=nonnegativity)



                else:
                    if search_radius_const is None:
                            search_radius_const = 10000

                    search_radius = search_radius_const * (float(step+1))**(-beta)/np.log(float(step+2))

                    # sparse code within radius
                    # error = self.compute_recons_error(data=X, loading=loading)
                    #print('recons_error_full ; ',error)
                    Code = self.update_code_within_radius(X_new_mat, W, U.T, r=search_radius,
                                                          alpha=0, subsample_ratio=subsample_ratio,
                                                          sub_iter = [5],
                                                          nonnegativity=nonnegativity)

                U_new = Code.T.reshape(U.shape)

                loading.update({'U' + str(mode): U_new})
                #print('!!! Iteration %i: %i th loading matrix updated..' % (step, mode))

            end = time.time()
            elapsed_time += end - start

            if if_compute_recons_error:
                error = self.compute_recons_error(data=X, loading=loading)
                # error *= normalization
                time_error = np.append(time_error, np.array([[elapsed_time, error]]), axis=0)
                print('!!! Reconstruction error at iteration %i = %f.3' % (step, error))


        result_dict.update({'loading': loading})
        # result_dict.update({'CPdict': self.out(loading)}) ### this is very expensive
        result_dict.update({'time_error': time_error.T})
        result_dict.update({'iter': iter})
        result_dict.update({'n_components': self.n_components})
        #np.save(save_folder + "/ALS_result_", result_dict)

        if output_results:
            return result_dict
        else:
            return loading



    def MU(self,
           iter=100,
           ini_loading=None,
           if_compute_recons_error=False,
           save_folder='Output_files',
           output_results=False):
        '''
        Given data tensor X and initial loading matrices W_ini, find loading matrices W by Multiplicative Update
        Ref: Shashua, Hazan, "Non-Negative Tensor Factorization with Applications to Statistics and Computer Vision" (2005)
        '''

        X = self.X

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', ini_loading.shape, '\n')

        n_modes = len(X.shape)
        if ini_loading is not None:
            loading = ini_loading.copy() # put .copy() Otherwise initial loading would change
        else:
            loading = {}
            for i in np.arange(X.ndim):
                loading.update({'U' + str(i): np.random.rand(X.shape[i], self.n_components)})

        result_dict = {}
        time_error = np.zeros(shape=[0, 2])
        elapsed_time = 0

        # Initial reconstruction error
        if if_compute_recons_error:
            error = self.compute_recons_error(data=self.X, loading=loading)
            time_error = np.append(time_error, np.array([[0, error]]), axis=0)
            print('!!! Reconstruction error at iteration %i = %f.3' % (0, error))

        for step in trange(int(iter)):
            start = time.time()

            for mode in np.arange(n_modes):
                # print('!!! X.shape', X.shape)
                X_new = np.swapaxes(X, mode, -1)
                U = loading.get('U' + str(mode))  # loading matrix to be updated
                loading_new = loading.copy()
                loading_new.update({'U' + str(mode): loading.get('U' + str(n_modes - 1))})
                loading_new.update({'U' + str(n_modes - 1): U})
                # Now update the last loading matrix U = 'U' + str(n_modes - 1)) by MU
                # Matrize X along the last mode to get a NMF problem V \approx W*H, and use MU in LEE & SEUNG (1999)

                # Form dictionary matrix
                CPdict = self.out(loading_new, drop_last_mode=True)
                # print('!!! X_new.shape', X_new.shape)
                W = np.zeros(shape=(len(X_new.reshape(-1, X_new.shape[-1])), self.n_components))
                for j in np.arange(self.n_components):
                    W[:, j] = CPdict.get('A' + str(j)).reshape(-1, 1)[:, 0]

                V = X_new.reshape(-1, X_new.shape[-1])
                # print('!!! W.shape', W.shape)
                # print('!!! U.shape', U.shape)
                # print('!!! V.shape', V.shape)
                #U_new = U.T * (W.T @ V) / (W.T @ W @ U.T)
                U_new = U * (V.T @ W) / (U @ W.T @ W)
                loading.update({'U' + str(mode): U_new})

                # print('!!! Iteration %i: %i th loading matrix updated..' % (step, mode))

            end = time.time()
            elapsed_time += end - start

            if if_compute_recons_error:
                error = self.compute_recons_error(data=self.X, loading=loading)
                time_error = np.append(time_error, np.array([[elapsed_time, error]]), axis=0)
                print('!!! Reconstruction error at iteration %i = %f.3' % (step, error))

        result_dict.update({'loading': loading})
        # result_dict.update({'CPdict': self.out(loading)}) ### this is very expensive
        result_dict.update({'time_error': time_error.T})
        result_dict.update({'iter': iter})
        result_dict.update({'n_components': self.n_components})
        #np.save(save_folder + "/MU_result_", result_dict)

        if output_results:
            return result_dict
        else:
            return loading


    def compute_recons_error(self, data, loading):

        CPdict = self.out(loading, drop_last_mode=False)
        recons = np.zeros(data.shape)
        for j in np.arange(len(CPdict.keys())):
            recons += CPdict.get('A' + str(j))
        error = np.linalg.norm((data - recons).reshape(-1, 1), ord=2)
        error /= np.linalg.norm(data)

        """
        # using matricization (equivalent to the above)
        X_new_mat = data.reshape(-1, data.shape[-1])
        CPdict = self.out(loading, drop_last_mode=True)
        H = loading.get('U' + str(self.n_modes))
        W = np.zeros(shape=(X_new_mat.shape[0], self.n_components))
        for j in np.arange(self.n_components):
            W[:, j] = CPdict.get('A' + str(j)).reshape(-1, 1)[:, 0]

        error = np.linalg.norm(X_new_mat - W @ H.T)
        error /= np.linalg.norm(X_new_mat)
        """

        return error
