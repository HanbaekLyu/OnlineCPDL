from utils.BCD_DR import ALS_DR
from utils.ocpdl import Online_CPDL
import numpy as np
import matplotlib.pyplot as plt
import pickle
from itertools import cycle
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def initialize_loading(data_shape=[10,10,10], n_components=5, scale=1):
    ### loading = python dict of [U1, U2, \cdots, Un], each Ui is I_i x R array
    loading = {}
    for i in np.arange(len(data_shape)):  # n_modes = X.ndim -1 where -1 for the last `batch mode'
        loading.update({'U' + str(i): scale*np.random.rand(data_shape[i], n_components)})
    return loading

def out(loading, drop_last_mode=False):
    ### given loading, take outer product of respected columns to get CPdict
    ### Use drop_last_mode for ALS
    CPdict = {}
    n_components = loading.get("U0").shape[1]
    for i in np.arange(n_components):
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

def Out_tensor(loading):
    ### given loading, take outer product of respected columns to get CPdict
    CPdict = out(loading, drop_last_mode=False)
    recons = np.zeros(CPdict.get('A0').shape)
    for j in np.arange(len(CPdict.keys())):
        recons += CPdict.get('A' + str(j))

    return recons


def ALS_run(X,
            n_components=10,
            iter=100,
            regularizer=None,  # L1 regularizer for each factor matrix
            ini_loading=None,
            beta=None,
            search_radius_const=1000,
            if_compute_recons_error=True,
            save_folder='Output_files',
            subsample_ratio = None,
            output_results=True):
    ALSDR = ALS_DR(X=X,
                        n_components=n_components,
                        ini_loading=None,
                        ini_A=None,
                        ini_B=None,
                        alpha=regularizer)

    result_dict = ALSDR.ALS(iter=iter,
                            ini_loading=ini_loading,
                            beta=beta,
                            search_radius_const=None,
                            if_compute_recons_error=if_compute_recons_error,
                            save_folder=save_folder,
                            subsample_ratio=subsample_ratio,
                            output_results=output_results)

    return result_dict


def MU_run(X,
           n_components=10,
           iter=100,
           regularizer=0,
           ini_loading=None,
           if_compute_recons_error=True,
           save_folder='Output_files',
           output_results=True):
    ALSDR = ALS_DR(X=X,
                        n_components=n_components,
                        ini_loading=None,
                        ini_A=None,
                        ini_B=None,
                        alpha=regularizer)

    result_dict = ALSDR.MU(iter=iter,
                           ini_loading=ini_loading,
                           if_compute_recons_error=if_compute_recons_error,
                           save_folder=save_folder,
                           output_results=output_results)
    return result_dict

def OCPDL_run(X,
              n_components=10,
              iter=100,
              regularizer=0,
              ini_loading=None,
              batch_size=100,
              beta=None,
              mode_2be_subsampled=-1,
              search_radius_const = None,
              if_compute_recons_error=True,
              save_folder='Output_files',
              output_results=True):
    OCPDL = Online_CPDL(X=X,
                        batch_size=batch_size,
                        iterations=iter,
                        n_components=n_components,
                        ini_loading=ini_loading,
                        ini_A=None,
                        ini_B=None,
                        alpha=regularizer,
                        beta=beta,
                        subsample=True)

    result_dict = OCPDL.train_dict(mode_2be_subsampled=mode_2be_subsampled,
                                   if_compute_recons_error=if_compute_recons_error,
                                   search_radius_const = search_radius_const,
                                   save_folder=save_folder,
                                   output_results=output_results)
    return result_dict

def plot_benchmark_errors(full_result_list, save_path):

    time_records = []
    errors = []
    f_interpolated_list = []

    # max duration and time records
    x_all_max = 0
    for i in np.arange(len(full_result_list)):
        errors0 = full_result_list[i].get('timed_errors_trials')
        x_all_max = max(x_all_max, max(errors0[:, :, -1][:, 0]))

    x_all = np.linspace(0, x_all_max, num=101, endpoint=True)

    for i in np.arange(len(full_result_list)):
        errors0 = full_result_list[i].get('timed_errors_trials')
        time_records.append(x_all[x_all < min(errors0[:, :, -1][:, 0])])

    # interpolate data and have common carrier
    for i in np.arange(len(full_result_list)):
        errors0 = full_result_list[i].get('timed_errors_trials')
        f0_interpolated = []
        for j in np.arange(errors0.shape[0]): # trials for same setting
            f0 = interp1d(errors0[j, 0, :], errors0[j, 1, :], fill_value="extrapolate")
            x_all_0 = time_records[i]
            f0_interpolated.append(f0(x_all_0))
        f0_interpolated = np.asarray(f0_interpolated)
        f_interpolated_list.append(f0_interpolated)

    # make figure
    search_radius_const = full_result_list[0].get('search_radius_const')
    color_list = ['g', 'k', 'r', 'c', 'b']
    marker_list = ['*', '|', 'x', 'o', '+']
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    for i in np.arange(len(full_result_list)):
        f0_interpolated = f_interpolated_list[i]
        f_avg0 = np.sum(f0_interpolated, axis=0) / f0_interpolated.shape[0]  ### axis-0 : trials
        f_std0 = np.std(f0_interpolated, axis=0)

        x_all_0 = time_records[i]
        color = color_list[i % len(color_list)]
        marker = marker_list[i % len(marker_list)]

        result_dict = full_result_list[i]
        beta = result_dict.get("beta")
        if (beta is None) and (result_dict.get("method")=="OCPDL"):
            label0 = result_dict.get("method") + " ($\\beta=$None)"
        elif (beta is None):
            label0 = result_dict.get("method")
        else:
            label0 = result_dict.get("method") + " ($\\beta=${}, $c'=${})".format(beta, search_radius_const)

        markers, caps, bars = axs.errorbar(x_all_0, f_avg0, yerr=f_std0,
                                           fmt=color+'-', marker=marker, label=label0, errorevery=5)
        axs.fill_between(x_all_0, f_avg0 - f_std0, f_avg0 + f_std0, facecolor=color, alpha=0.1)

    # min_max duration
    x_all_min_max = []
    for i in np.arange(len(time_records)):
        x_all_ALS0 = time_records[i]
        x_all_min_max.append(max(x_all_ALS0))

    x_all_min_max = min(x_all_min_max)
    axs.set_xlim(0, x_all_min_max)


    [bar.set_alpha(0.5) for bar in bars]
    # axs.set_ylim(0, np.maximum(np.max(f_OCPDL_avg + f_OCPDL_std), np.max(f_ALS_avg + f_ALS_std)) * 1.1)
    axs.set_xlabel('Elapsed time (s)', fontsize=13)
    axs.set_ylabel('Relative recons. error', fontsize=13)
    data_name = full_result_list[0].get('data_name')
    title = data_name
    plt.suptitle(title, fontsize=13)
    axs.legend(fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.00, 0.00)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    plt.savefig(save_path)


def main(n_components = 5,
        iter = 30,
        num_repeat = 10,
        save_folder = "Output_files/JMLR_OCPDL3",
        data_name = "Synthetic", # "Synthetic" or "Twitter"
        run_ALS = True,
        run_MU = True,
        run_OCPDL = True,
        plot_errors = True):



    # Load data
    loading = {}

    # Load data
    if data_name == "Synthetic":
        scale = 0.0001
        np.random.seed(1)
        #U0 = np.random.rand(20, 1*n_components)
        U0 = np.random.randint(2,size=[100, 2*n_components])
        np.random.seed(2)
        U1 = np.random.rand(100, 2*n_components)
        np.random.seed(3)
        U2 = np.random.rand(5000, 2*n_components)

        loading.update({'U0': U0})
        loading.update({'U1': U1})
        loading.update({'U2': U2})

        X = Out_tensor(loading) * scale

    elif data_name == "Twitter":
        path = "Data/Twitter/top_1000_daily/data_tensor_top1000.pickle"
        dict = pickle.load(open(path, "rb"))
        X = dict[1]
        file_name = "Twitter"
        #X = np.swapaxes(X, 1, 2)

    elif data_name == "Headlines":
        path = "Data/headlines_tensor.pickle"
        dict = pickle.load(open(path, "rb"))
        X = dict
        file_name = "Headlines"

    elif data_name == "20Newsgroups":
        path = "Data/20news_tfidf_tensor.pickle"
        dict = pickle.load(open(path, "rb"))
        X = dict * 100
        file_name = "20Newsgroups"




    print('X.shape', X.shape)

    # X = 10 * X/np.linalg.norm(X)

    search_radius_const = 10
    # search_radius_const = int(np.linalg.norm(X.reshape(-1,1),1))
    print('search_radius_const', search_radius_const)

    loading_list = []
    scale_X = 1/np.linalg.norm(X.reshape(-1,1))
    for i in np.arange(num_repeat):
        loading_list.append(initialize_loading(data_shape=X.shape, n_components=n_components, scale=scale_X))

    full_result_list = []

    if run_ALS:
        # beta_list = [1 / 2, 1, None]
        beta_list = [None]
        for beta in beta_list:
            print('!!! ALS initialized with beta:', beta)
            list_full_timed_errors = []
            results_dict = {}
            for i in np.arange(num_repeat):

                result_dict_ALS = ALS_run(X,
                                          n_components=n_components,
                                          iter=iter,
                                          regularizer=0,
                                          # inverse regularizer on time mode (to promote long-lasting topics),
                                          # no regularizer on on words and tweets
                                          ini_loading=loading_list[i],
                                          beta=beta,
                                          search_radius_const=None,
                                          subsample_ratio=None,
                                          if_compute_recons_error=True,
                                          save_folder=save_folder,
                                          output_results=True)
                time_error = result_dict_ALS.get('time_error')
                list_full_timed_errors.append(time_error.copy())
                # print('!!! list_full_timed_errors', len(list_full_timed_errors))

            timed_errors_trials = np.asarray(
                list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)

            if beta is None:
                results_dict.update({"method": "ALS"})
            else:
                results_dict.update({"method": "BCD-DR"})
            results_dict.update({"data_name": data_name})
            results_dict.update({"beta": beta})
            results_dict.update({"search_radius_const": search_radius_const})
            results_dict.update({"n_components": n_components})
            results_dict.update({"iterations": iter})
            results_dict.update({"num_trials": num_repeat})
            results_dict.update({'timed_errors_trials': timed_errors_trials})
            full_result_list.append(results_dict.copy())

            save_path = save_folder + "/full_result_list_" + str(data_name)
            np.save(save_path, full_result_list)
            # print('full_result_list:', full_result_list)

    if run_MU:
        results_dict = {}
        print('!!! MU initialized')
        list_full_timed_errors = []
        iter1 = iter
        for i in np.arange(num_repeat):
            result_dict_MU = MU_run(X,
                                    n_components=n_components,
                                    iter=iter1,
                                    regularizer=0,
                                    ini_loading=loading_list[i],
                                    if_compute_recons_error=True,
                                    save_folder=save_folder,
                                    output_results=True)
            time_error = result_dict_MU.get('time_error')
            list_full_timed_errors.append(time_error.copy())
            # print('!!! list_full_timed_errors', len(list_full_timed_errors))

        timed_errors_trials = np.asarray(
            list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)

        results_dict.update({"method": "MU"})
        results_dict.update({"data_name": data_name})
        results_dict.update({"n_components": n_components})
        results_dict.update({"search_radius_const": search_radius_const})
        results_dict.update({"iterations": iter1})
        results_dict.update({"num_trials": num_repeat})
        results_dict.update({'timed_errors_trials': timed_errors_trials})
        full_result_list.append(results_dict.copy())

        save_path = save_folder + "/full_result_list_" + str(data_name)
        np.save(save_path, full_result_list)
        # print('full_result_list:', full_result_list)

    if run_OCPDL:
        beta_list = [3/4, 1, None]
        #beta_list = [3/4]
        for beta in beta_list:
            print('!!! OCPDL initialized with beta:', beta)
            list_full_timed_errors = []
            results_dict = {}
            for i in np.arange(num_repeat):

                result_dict_OCPDL = OCPDL_run(X,
                                          n_components=n_components,
                                          iter=iter,
                                          regularizer=0,
                                          batch_size = int(X.shape[-1]/5),
                                          #batch_size = 10,
                                          # inverse regularizer on time mode (to promote long-lasting topics),
                                          # no regularizer on on words and tweets
                                          ini_loading=loading_list[i].copy(),
                                          beta=beta,
                                          search_radius_const=search_radius_const,
                                          if_compute_recons_error=True,
                                          #mode_2be_subsampled=1,
                                          save_folder=save_folder,
                                          output_results=True)
                time_error = result_dict_OCPDL.get('time_error')
                print('!!! time_error', time_error)
                list_full_timed_errors.append(time_error.copy())
                # print('!!! list_full_timed_errors', len(list_full_timed_errors))

            timed_errors_trials = np.asarray(
                list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)

            results_dict.update({"method": "OCPDL"})
            results_dict.update({"data_name": data_name})
            results_dict.update({"beta": beta})
            results_dict.update({"search_radius_const": search_radius_const})
            results_dict.update({"n_components": n_components})
            results_dict.update({"iterations": iter})
            results_dict.update({"num_trials": num_repeat})
            results_dict.update({'timed_errors_trials': timed_errors_trials})
            full_result_list.append(results_dict.copy())

            save_path = save_folder + "/full_result_list_" + str(data_name)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.save(save_path, full_result_list)
            # print('full_result_list:', full_result_list)


    if plot_errors:
        full_result_list = np.load(save_folder + "/full_result_list_" + str(data_name) + ".npy", allow_pickle=True)
        full_result_list = full_result_list[::-1]
        n_trials = full_result_list[0].get("num_trials")
        n_components = full_result_list[0].get('n_components')
        search_radius_const = full_result_list[0].get('search_radius_const')
        data_name = full_result_list[0].get('data_name')

        save_path = save_folder + "/full_result_error_plot" + '_ntrials_' + str(n_trials) + "_" + "_ncomps_" + str(
            n_components) + "_src_" + str(search_radius_const) + "_" + str(data_name) + ".pdf"

        plot_benchmark_errors(full_result_list, save_path=save_path)
        print('!!! plot saved')


if __name__ == '__main__':
    main()
