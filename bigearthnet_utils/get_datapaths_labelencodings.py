# ==============================================================================
# Desc : Contains functions to create a csv file containing data paths to data folders (image and labels)
# and using encordings to convert string labels to numerical format.
# ==============================================================================
import numpy as np
import pandas as pd 
import json 
import pickle
import os
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Tuple
from tqdm import tqdm
from termcolor import colored

def construct_dataframe(dataset_path:str):
    """
    Desc : Constructs a pandas dataframe containing the path to images and labels
    Inputs :
        - dataset_path : root directory contianing all folders of images and labels
    """
    df = pd.DataFrame({"data_folder_path" : [], "red_band_path" : [], "green_band_path" : [], "blue_band_path" : [], "raw_labels" : []})
    print(colored(df.shape, "green"))
    for data_folder in tqdm(os.listdir(dataset_path)):
        # Path containing to folder containing 12 tif files and a json file 
        data_folder_path = os.path.join(dataset_path, data_folder)
        
        # Path to RGB channels | rb refers to red band ...
        rb_path = os.path.join(data_folder_path, [f for f in os.listdir(data_folder_path) if f.endswith("B04.tif")][0])
        gb_path = os.path.join(data_folder_path, [f for f in os.listdir(data_folder_path) if f.endswith("B03.tif")][0])
        bb_path = os.path.join(data_folder_path, [f for f in os.listdir(data_folder_path) if f.endswith("B02.tif")][0])

        # Path to json file
        json_path = os.path.join(data_folder_path, [f for f in os.listdir(data_folder_path) if f.endswith(".json")][0])

        # Read json file to get labels
        with open(json_path) as f:
            meta_data = json.load(f)

        # Append into pandas dataframe
        new_row = { 
            "red_band_path" : rb_path,
            "green_band_path" : gb_path,
            "blue_band_path" : bb_path,
            "raw_labels" : meta_data["labels"]
        }

        df.loc[len(df)] = [data_folder_path, rb_path, gb_path, bb_path, str(meta_data["labels"])]
    
    # Set index to be the path to data folders
    df.set_index("data_folder_path", inplace = True)

    return df

def convert_labels_to_encodings(df:pd.DataFrame)->Tuple[pd.DataFrame, MultiLabelBinarizer]:
    """
    Desc : Converts labels in dataframe to encordings
    Inputs
        - df : pandas dataframe containing raw_labels in string format 
    Outputs
        - df : pandas dataframe containing 
        - mlb : Trained MultiLabelBinarizer
    """
    # Convert String Labels to Encodings
    str_labels = list(map(lambda x : ast.literal_eval(x), df["raw_labels"]))
    mlb = MultiLabelBinarizer()
    enc_labels = mlb.fit_transform(str_labels)

    # Convert the encoded labels back to a string format to save it as a list in pandas dataframe
    enc_labels_str_format = list(map(lambda x: str(x.tolist()), enc_labels))
    df["enc_labels"] = enc_labels_str_format

    return df, mlb

if __name__ == "__main__":

    # Loop through every folder in BigEarthNet and save, paths to color channels, paths to root data folder and class labels 
    dataset_path = "data/BigEarthNet-v1.0"
    df = construct_dataframe(dataset_path)
    df.to_csv("bigearthnet_utils/ben_datapaths_and_labels_v3.csv", index = False)

    # # Load the saved csv file containing file paths to RGB Channels, folder and string labels
    # #! File moved to archive as its not useful anymore - A more updated file ben_datapaths_and_labs_v2.csv containing the encodings is available
    df = pd.read_csv("bigearthnet_utils/ben_datapaths_and_labels_v3.csv")

    # Add label encodings to dataframe
    df, mlb = convert_labels_to_encodings(df)

    # Save dataframe and multilabel binarizer for further use
    df.to_csv("bigearthnet_utils/ben_datapaths_and_labels_v4.csv")
    with open("bigearthnet_utils/ben_mlb_encoder_labels_v4.pkl", "wb") as f:
        pickle.dump(mlb, f)

    # Load the saved csv file containing file paths to RGB Channels, folder, string labels and encoded labels
    # df = pd.read_csv("bigearthnet_utils/ben_datapaths_and_labels_v2.csv")

    # Test trained multilabel bianrizer
    # with open("bigearthnet_utils/ben_mlb_encoder_labels.pkl", "rb") as f:
    #     mlb = pickle.load(f)

    # enc_labs = np.array(list(map(lambda x: ast.literal_eval(x), df["enc_labels"])))
    # str_labs = mlb.inverse_transform(enc_labs)

    # print(df.head())
    # print(str_labs[:5])

