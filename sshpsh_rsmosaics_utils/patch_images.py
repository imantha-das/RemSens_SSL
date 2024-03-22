# ==============================================================================
# Desc : Script to patch large RS images into small portions
# ==============================================================================

import os 
from os import path 
from glob import glob 
import numpy as np
from PIL import Image
import skimage as ski
import matplotlib.pyplot as plt
from patchify import patchify

#To avoid Image.DecompressionBombError
Image.MAX_IMAGE_PIXELS = 1000000000

def patch_image(path_to_img:str, patch_size = 512,resize = None):
    img = Image.open(path_to_img)
    W,H = img.size
    if resize:
        resimg = img.resize(resize)
        resimg_arr = np.asarray(resimg)
        assert resimg_arr.ndim == 3
        w,h,c = resimg_arr.shape
        print(f"Resizing image of shape : {np.asarray(img).shape} -> {resimg_arr.shape}")
        # patches.shape = (15, 15, 1, 512, 512, 3)
        patches = patchify(resimg_arr, patch_size = (patch_size, patch_size, c), step = patch_size)
    else:
        img_arr = np.asarray(img)
        assert img_arr.ndim == 3
        w,h,c = img_arr.shape
        patches = patchify(img_arr, patch_size = (patch_size, patch_size, c), step = patch_size)
    
    patch_list  = []
    for p_row in range(patches.shape[0]):
        for p_col in range(patches.shape[1]):
            patch = patches[p_row,p_col,0,:,:,:]
            patch_list.append(patch)
        
 
    return patch_list, img

def plot_patches(patch_list):
    rows = int(np.sqrt(len(patch_list)))
    cols = rows

    fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (10,10))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax[r,c].imshow(patch_list[idx])
            ax[r,c].axis("off")
            idx += 1

    plt.show()

def convert_and_save(arr, f_prefix, idx):
    img =Image.fromarray(arr)
    fname = f"{f_prefix}_{idx}.tif"
    print(f"saving image to : {fname}")
    img.save(fname)

if __name__ == "__main__":

    # Folder where the images are stored
    root = "data/SSHSPH-RSMosaics-MY-v2.1/images/channel3"
    # Parent folder of the folder contianing the images - where you will store the patchse
    parent_folder = path.dirname(root)
    save_folder = path.join(parent_folder, "channel3_p")
    # Creare a folder to store the patches
    if not path.exists(save_folder):
        os.mkdir(save_folder)

    # Loop throught the folder containing all the images
    for folder in glob(path.join(root, "*")):
        # Loop through each of the files in the image folder ( there are many including the tif ...)
        for file in glob(path.join(folder, "*")):
            if file.endswith(".tif"):
                # Get the patches list and orginal image (if you want to have a look)
                patch_list, img = patch_image(file, resize = (8000,8000))
                # Save patches in the designated folder
                f_prefix = path.basename(path.dirname(file)) # get the foldername, i.w 20140429_03
                f_prefix = path.join(save_folder, f_prefix) # add path to begining, "data/SSHSPH.../test_ch3_p/20140429_03"
                print(f"file name prefix : {f_prefix}")
                for i in range(len(patch_list)):
                    convert_and_save(patch_list[i], f_prefix, i)

        

