from utils.image_reconstruction_OCPDL import Image_Reconstructor_OCPDL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def main():

    patch_size = 20
    # sources = ["Data/monet/monet" + str(n) + ".png" for n in np.arange(0,1)]
    # sources = ["Data/picasso/picasso" + str(n) + ".jpg" for n in np.arange(0, 1)]
    # sources = ["Data/gogh/gogh" + str(n) + ".jpg" for n in np.arange(0, 1)]
    sources = ["Data/classic_paintings/gogh_cafe.jpg"]
    for path in sources:
        reconstructor = Image_Reconstructor_OCPDL(path=path,
                                                   n_components=18,  # number of dictionary elements -- rank
                                                   iterations=40,  # number of iterations for the ONTF algorithm
                                                   sub_iterations=2, # number of i.i.d. subsampling for each iteration of ONTF
                                                   batch_size=100,  # number of patches used in i.i.d. subsampling
                                                   num_patches=100, # number of patches that ONTF algorithm learns from at each iteration
                                                   sub_num_patches=5000, # number of patches to optimize H after training W
                                                   downscale_factor=1,
                                                   patch_size=patch_size,
                                                   patches_file='',
                                                   alpha = 1,
                                                   is_matrix=False,
                                                   unfold_space=False,
                                                   unfold_all=True,
                                                   is_color=True)

        train_fresh = True

        if train_fresh:
            W = reconstructor.train_dict()
            CPdict = reconstructor.out(W)
            print('W', W)
            print('W.keys()', W.keys())
            print('CPdict.keys()', CPdict.keys())
            print('U0.shape', W.get('U0').shape)
            print('U1.shape', W.get('U1').shape)

        display_dictionary = True

        if display_dictionary and train_fresh:
            reconstructor.display_dictionary_CP(W, plot_shape_N_color=False)
        else:
            path = 'Image_dictionary/dict_learned_CPDL_gogh_allfold.npy'
            loading = np.load(path, allow_pickle=True).item()
            reconstructor.display_dictionary_CP(loading, plot_shape_N_color=False)


        if_reconstruct = False

        if if_reconstruct:
            path = 'Image_dictionary/dict_learned_CPDL_piccaso.npy'
            loading = np.load(path, allow_pickle=True).item()
            IMG_recons = reconstructor.reconstruct_image_color(loading=loading, recons_resolution=5, if_save=False)

        # reconstructor.reconstruct_image("Data/escher/10.jpg", downscale_factor=1, patch_size=patch_size, is_matrix=False)
        # reconstructor.reconstruct_image("Data/renoir/0.jpg", downscale_factor=1, patch_size=patch_size, is_matrix=False)


if __name__ == '__main__':
    main()

