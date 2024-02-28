# ==============================================================================
# Desc : Explore Dataset such as identify the size of each image
# ==============================================================================
import os 
from glob import glob
import pandas as pd
import subprocess
import skimage as ski

def get_image_size(img_paths:str)->dict:
    """Returns a pandas dataframe with path to image and size"""
    # Dictionary to keep track of image paths and shape of image
    img_paths_shape = {}
    # Loop through all the folders contianing images
    for folder in img_paths:
        # Loop through the contents in each image folder
        tif_count = 0
        for file in glob(folder + "/" + "*"):
            if file.endswith("tif"):
                img = ski.io.imread(file)
                img_paths_shape[file] = img.shape
                tif_count += 1
        assert tif_count == 1, f"More or less than 1 tif file, tif_count : {tif_count}"

    return img_paths_shape

if __name__ == "__main__":

    parent_dir = "data/SSHSPH-RSMosaics-MY-v2.1/tiles/channel4"
    image_folders = glob(os.path.join(parent_dir,"*"))

    img_shp = get_image_size(image_folders)
    img_shp_df = pd.DataFrame(columns = ["path","shape"])
    img_shp_df["path"] = img_shp.keys()
    img_shp_df["shape"] = img_shp.values()
    img_shp_df.to_csv("sshpsh_rsmosaics_utils/rsmosaics_v2.1_tiles_ch4_shps.csv", index = False)
        
