from utils.ocpdl import Online_CPDL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d


def Out_tensor(loading):
    ### given loading, take outer product of respected columns to get CPdict
    CPdict = {}
    n_modes = len(loading.keys())
    n_components = loading.get('U0').shape[1]
    print('!!! n_modes', n_modes)
    print('!!! n_components', n_components)

    for i in np.arange(n_components):
        A = np.array([1])
        for j in np.arange(n_modes):
            loading_factor = loading.get('U' + str(j))  ### I_i X n_components matrix
            # print('loading_factor', loading_factor)
            A = np.multiply.outer(A, loading_factor[:, i])
        A = A[0]
        CPdict.update({'A' + str(i): A})
    print('!!! CPdict.keys()', CPdict.keys())

    X = np.zeros(shape=CPdict.get('A0').shape)
    for j in np.arange(len(loading.keys())):
        X += CPdict.get('A' + str(j))

    return X


def ALS_run(X,
            n_components=10,
            iter=100,
            regularizer=0,
            ini_loading=None,
            if_compute_recons_error=True,
            save_foler='Output_files',
            output_results=True):
    OCPDL = Online_CPDL(X=X,
                        n_components=n_components,
                        ini_loading=None,
                        ini_A=None,
                        ini_B=None,
                        alpha=regularizer)

    result_dict = OCPDL.ALS(iter=iter,
                            ini_loading=ini_loading,
                            if_compute_recons_error=if_compute_recons_error,
                            save_folder=save_foler,
                            output_results=output_results)
    return result_dict


def MU_run(X,
           n_components=10,
           iter=100,
           regularizer=0,
           ini_loading=None,
           if_compute_recons_error=True,
           save_foler='Output_files',
           output_results=True):
    OCPDL = Online_CPDL(X=X,
                        n_components=n_components,
                        ini_loading=None,
                        ini_A=None,
                        ini_B=None,
                        alpha=regularizer)

    result_dict = OCPDL.MU(iter=iter,
                           ini_loading=ini_loading,
                           if_compute_recons_error=if_compute_recons_error,
                           save_folder=save_foler,
                           output_results=output_results)
    return result_dict


def OCPDL_run(X,
              n_components=10,
              iter=100,
              regularizer=0,
              ini_loading=None,
              batch_size=100,
              mode_2be_subsampled=-1,
              if_compute_recons_error=True,
              save_foler='Output_files',
              output_results=True):
    OCPDL = Online_CPDL(X=X,
                        batch_size=batch_size,
                        iterations=iter,
                        n_components=n_components,
                        ini_loading=ini_loading,
                        ini_A=None,
                        ini_B=None,
                        alpha=regularizer,
                        subsample=True)

    result_dict = OCPDL.train_dict(mode_2be_subsampled=mode_2be_subsampled,
                                   if_compute_recons_error=if_compute_recons_error,
                                   save_folder=save_foler,
                                   output_results=output_results)
    return result_dict


def plot_benchmark_errors(ALS_result, OCPDL_result, name=1, errorbar=True):
    n_components = ALS_result.get('n_components')

    if not errorbar:
        ALS_errors = ALS_result.get('time_error')
        OCLDP_errors = OCPDL_result.get('time_error')

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        axs.plot(ALS_errors[0, :], ALS_errors[1, :], label='ALS')
        axs.plot(OCLDP_errors[0, :], OCLDP_errors[1, :], label='OCPDL')
        axs.set_xlabel('Elapsed time (s)')
        axs.set_ylabel('Reconstruction error')
        plt.suptitle('Reconstruction error benchmarks')
        axs.legend()
        plt.tight_layout()
        plt.suptitle('Reconstruction error benchmarks', fontsize=13)
        plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.00, 0.00)
        plt.savefig('Output_files/benchmark_plot' + '_' + str(name))
        # plt.show()
    else:
        ALS_errors = ALS_result.get('timed_errors_trials')  # shape (# trials) x (2 for time, error) x (iterations)
        OCPDL_errors = OCPDL_result.get('timed_errors_trials')
        n_trials = ALS_errors.shape[0]
        x_all = np.linspace(0, min(OCPDL_errors[:, :, -1][:, 0]), num=101,
                            endpoint=True)  # ends at the avg of last times of OCPDL runs
        print('!!! x_all', x_all)
        # interpolate data and have common carrier

        f_ALS_interpolated = []
        f_OCPDL_interpolated = []
        for i in np.arange(n_trials):
            f_ALS = interp1d(ALS_errors[i, 0, :], ALS_errors[i, 1, :], fill_value="extrapolate")
            f_ALS_interpolated.append(f_ALS(x_all))
            f_OCPDL = interp1d(OCPDL_errors[i, 0, :], OCPDL_errors[i, 1, :], fill_value="extrapolate")
            f_OCPDL_interpolated.append(f_OCPDL(x_all))

        f_ALS_interpolated = np.asarray(f_ALS_interpolated)
        f_OCPDL_interpolated = np.asarray(f_OCPDL_interpolated)

        f_ALS_avg = np.sum(f_ALS_interpolated, axis=0) / f_ALS_interpolated.shape[0]  ### axis-0 : trials
        f_ALS_std = np.std(f_ALS_interpolated, axis=0)
        print('!!! f_ALS_std', f_ALS_std)
        f_OCPDL_avg = np.sum(f_OCPDL_interpolated, axis=0) / f_OCPDL_interpolated.shape[0]  ### axis-0 : trials
        f_OCPDL_std = np.std(f_OCPDL_interpolated, axis=0)
        print('!!! f_OCPDL_std', f_OCPDL_std)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        markers, caps, bars = axs.errorbar(x_all, f_ALS_avg, yerr=f_ALS_std,
                                           fmt='r-', label='ALS', errorevery=5)
        markers, caps, bars = axs.errorbar(x_all, f_OCPDL_avg, yerr=f_OCPDL_std,
                                           fmt='b-', label='OCPDL', errorevery=5)
        [bar.set_alpha(0.5) for bar in bars]
        axs.set_ylim(0, np.maximum(np.max(f_OCPDL_avg + f_OCPDL_std), np.max(f_ALS_avg + f_ALS_std)) * 1.1)
        axs.set_xlabel('Elapsed time (s)')
        axs.set_ylabel('Reconstruction error')
        plt.suptitle('Reconstruction error benchmarks')
        axs.legend()
        plt.tight_layout()
        plt.suptitle('Reconstruction error benchmarks', fontsize=13)
        plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.00, 0.00)
        plt.savefig('Output_files/benchmark_plot_errorbar' + '_ntrials_' + str(n_trials) + "_" + "_ncomps_" + str(
            n_components) + "_" + str(name))


def main():
    loading = {}
    n_components = 5
    iter = 20
    num_repeat = 10
    save_folder = "Output_files"

    synthetic_data = True
    run_ALS = False
    run_MU = True
    run_OCPDL = True
    plot_errors = True
    file_identifier = 'new1'

    # Load data
    file_name = "Synthetic"
    if synthetic_data:
        np.random.seed(1)
        U0 = np.random.rand(10, n_components)
        np.random.seed(2)
        U1 = np.random.rand(10, n_components)
        np.random.seed(3)
        U2 = np.random.rand(1000, n_components)

        loading.update({'U0': U0})
        loading.update({'U1': U1})
        loading.update({'U2': U2})
        X = Out_tensor(loading)
    else:
        path = "Data/Twitter/top_1000_daily/data_tensor_top1000.pickle"
        dict = pickle.load(open(path, "rb"))
        X = dict[1]
        file_name = "Twitter"

    file_name = file_name + "_" + file_identifier

    print('X.shape', X.shape)

    if run_ALS:
        list_full_timed_errors = []
        for i in np.arange(num_repeat):
            result_dict_ALS = ALS_run(X,
                                      n_components=n_components,
                                      iter=iter,
                                      regularizer=0,
                                      ini_loading=None,
                                      if_compute_recons_error=True,
                                      save_foler='Output_files',
                                      output_results=True)
            time_error = result_dict_ALS.get('time_error')
            list_full_timed_errors.append(time_error.copy())
            print('!!! list_full_timed_errors', len(list_full_timed_errors))

        timed_errors_trials = np.asarray(
            list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)
        result_dict_ALS.update({'timed_errors_trials': timed_errors_trials})

        np.save(save_folder + "/ALS_result_" + str(file_name), result_dict_ALS)
        print('result_dict_ALS.keys()', result_dict_ALS.keys())

    if run_MU:
        list_full_timed_errors = []
        for i in np.arange(num_repeat):
            result_dict_MU = MU_run(X,
                                    n_components=n_components,
                                    iter=iter,
                                    regularizer=0,
                                    ini_loading=None,
                                    if_compute_recons_error=True,
                                    save_foler='Output_files',
                                    output_results=True)
            time_error = result_dict_MU.get('time_error')
            list_full_timed_errors.append(time_error.copy())
            print('!!! list_full_timed_errors', len(list_full_timed_errors))

        timed_errors_trials = np.asarray(
            list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)
        result_dict_ALS.update({'timed_errors_trials': timed_errors_trials})

        np.save(save_folder + "/MU_result_" + str(file_name), result_dict_ALS)
        print('result_dict_MU.keys()', result_dict_ALS.keys())

    if run_OCPDL:
        list_full_timed_errors = []
        for i in np.arange(num_repeat):
            result_dict_OCPDL = OCPDL_run(X,
                                          n_components=n_components,
                                          iter=iter,
                                          regularizer=0,
                                          ini_loading=None,
                                          mode_2be_subsampled=-1,
                                          if_compute_recons_error=True,
                                          save_foler='Output_files',
                                          output_results=True)

            time_error = result_dict_OCPDL.get('time_error')
            list_full_timed_errors.append(time_error.copy())

        timed_errors_trials = np.asarray(
            list_full_timed_errors)  # shape (# trials) x (2 for time, error) x (iterations)
        result_dict_OCPDL.update({'timed_errors_trials': timed_errors_trials})
        print('!!! list_full_timed_errors', len(list_full_timed_errors))

        np.save(save_folder + "/OCPDL_result_" + str(file_name), result_dict_OCPDL)
        print('result_dict_OCPDL.keys()', result_dict_OCPDL.keys())

    if plot_errors:
        save_filename = file_name + ".npy"
        if synthetic_data:
            ALS_result_Synthetic = np.load('Output_files/ALS_result_' + save_filename, allow_pickle=True).item()
            OCPDL_result_Synthetic = np.load('Output_files/OCPDL_result_' + save_filename, allow_pickle=True).item()
            MU_result_Synthetic = np.load('Output_files/MU_result_' + save_filename, allow_pickle=True).item()
            plot_benchmark_errors(ALS_result_Synthetic, OCPDL_result_Synthetic, name=file_name, errorbar=True)

        else:
            ALS_result_Twitter = np.load('Output_files/ALS_result_' + save_filename, allow_pickle=True).item()
            OCPDL_result_Twitter = np.load('Output_files/OCPDL_result_' + save_filename, allow_pickle=True).item()
            MU_result_Synthetic = np.load('Output_files/MU_result_' + save_filename, allow_pickle=True).item()
            plot_benchmark_errors(ALS_result_Twitter, OCPDL_result_Twitter, name=file_name, errorbar=True)


if __name__ == '__main__':
    main()

