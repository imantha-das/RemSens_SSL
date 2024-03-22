from qgis.core import QgsProject, QgsRasterLayer
from glob import glob 
from os import path 
import os 

def load_tif_to_qgis(root,layer_suffix = "_tile_ch3"):
    # Make an instance 
    project = QgsProject.instance()

    # Loop through all the folders 
    for folder in glob(path.join(root, "*")):
        # Loop through all the files in the folders
        for file in glob(path.join(folder, "*")):
            # Locate the tif file
            if file.endswith(".tif"):
                # give the layer a name which is the folder name
                layer_name = path.basename(path.dirname(file)) # get parent folder name where file is stored
                layer_name = layer_name + layer_suffix
                layer = QgsRasterLayer(file, layer_name, "gdal")
                if layer.isValid():
                    project.addMapLayer(layer)


root = "/home/imantha/workspace/RemSens_SSL/data/SSHSPH-RSMosaics-MY-v2.1/images/channel3"
    #root = "/home/imantha/workspace/RemSens_SSL/data/SSHSPH-RSMosaics-MY-v2.1/tiles"

load_tif_to_qgis(root, layer_suffix = "")
    
            