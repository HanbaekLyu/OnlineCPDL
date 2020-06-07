from utils.video_reconstruction_OCPDL import Video_Reconstructor_OCPDL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def main():

    segment_length = 1
    sources = ["Data/Video/mouse_brain_activity.mp4"]
    # sources = ["Data/Video/monet" + str(n) + ".png" for n in np.arange(0,1)]
    # sources = ["Data/picasso/picasso" + str(n) + ".jpg" for n in np.arange(0, 1)]
    for path in sources:
        reconstructor = Video_Reconstructor_OCPDL(path=path,
                                                   n_components=25,  # number of dictionary elements -- rank
                                                   iterations=10,  # number of iterations for the ONTF algorithm
                                                   sub_iterations=2, # number of i.i.d. subsampling for each iteration of ONTF
                                                   batch_size=1,  # number of patches used in i.i.d. subsampling
                                                   num_patches=1, # number of patches that ONTF algorithm learns from at each iteration
                                                   downscale_factor=1,
                                                   segment_length=segment_length,
                                                   patches_file='',
                                                   is_matrix=False,
                                                   unfold_space=False,
                                                   unfold_all=True,
                                                   is_color=True)

        train_fresh = False

        if train_fresh:
            W = reconstructor.train_dict()
            CPdict = reconstructor.out(W)
            print('W', W)
            print('W.keys()', W.keys())
            print('CPdict.keys()', CPdict.keys())
            print('U0.shape', W.get('U0').shape)
            print('U1.shape', W.get('U1').shape)
        else:
            path = 'Video_dictionary/dict_learned_CPDL_iter100.npy'
            W = np.load(path, allow_pickle=True).item()

        display_dictionary = True

        if display_dictionary:
            path = 'Video_dictionary/dict_learned_CPDL_iter100.npy'
            W = np.load(path, allow_pickle=True).item()
            reconstructor.display_dictionary_CP(W, plot_shape_N_color=False)


        if_reconstruct = False

        if if_reconstruct:
            path = 'Image_dictionary/dict_learned_CPDL_klimpt_allfold.npy'
            loading = np.load(path, allow_pickle=True).item()
            IMG_recons = reconstructor.reconstruct_image_color(loading=loading, recons_resolution=5, if_save=False)

        # reconstructor.reconstruct_image("Data/escher/10.jpg", downscale_factor=1, patch_size=patch_size, is_matrix=False)
        # reconstructor.reconstruct_image("Data/renoir/0.jpg", downscale_factor=1, patch_size=patch_size, is_matrix=False)


if __name__ == '__main__':
    main()