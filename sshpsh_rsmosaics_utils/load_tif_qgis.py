from qgis.core import QgsProject, QgsRasterLayer
from glob import glob 
from os import path 
import os 

root = "workspace/RemSens_SSL/data/SSHSPH-RSMosaics-MY-v2.1/tiles/channel4"
#root = "/home/imantha/workspace/RemSens_SSL/data/SSHSPH-RSMosaics-MY-v2.1/tiles"

# Make an instance 
project = QgsProject.instance()

for folder in glob(path.join(root, "*")):
    for file in glob(path.join(folder, "*")):
        if file.endswith(".tif"):
            layer_name = path.basename(path.dirname(file)) # get parent folder name where file is stored
            layer_name = layer_name + "_tiles_ch4"
            layer = QgsRasterLayer(file, layer_name, "gdal")
            if layer.isValid():
                project.addMapLayer(layer)
    
            