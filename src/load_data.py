import sys
import os
import json
import warnings

import datetime
import numpy as np
import logging
import re
import requests
import argparse
import glob
import yaml
import pandas as pd

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')



def load_csv(input_data_path, columns):
    """Function to get data as a dataframe from a path given.
    Args:
        input_data_path (str): Location where the csv file is saved.
        columns (:obj:`list`): List of features to extract.
    Returns:
        df (:py:class:`pandas.DataFrame`): DataFrame with specific features and target.
    """
    
    # load data
    df = pd.read_csv(input_data_path)

    return df[columns]

def load_s3(url, file_name, save_path, **kwarges):
    """Function to download data from the s3 public bucket"""

    try:
        r = requests.get(url)
        logger.info("Download %s from bucket url %s", file_name, url)
        open(save_path, 'wb').write(r.content)
    except requests.exceptions.RequestException:
        logger.error("Error: Unable to download the file %s", file_name)
        print('please check the name of file or the valid url or a given saved_path')



def load_data(config):
    """Function to get data as a dataframe from a csv file.
    Returns:
        df (:py:class:`pandas.DataFrame`): DataFrame containing features and labels.
    """
    if 'load_s3' in config:
        load_s3(**config['load_s3'])
    else: 
        raise ValueError('No data is loaded from s3')


    how = config["how"].lower()

    if how == "load_csv":
        if "load_csv" not in config:
            raise ValueError("'how' given as 'load_csv' but 'load_csv' not in configuration")
        else:
            df = load_csv(**config["load_csv"]) # load dataframe with all columns selected
    else:
        raise ValueError("Option for 'how' is 'load_csv' but %s was given" % how)
    
    # save data
    if "save_data" in config and config["save_data"] is not None:
        df.to_csv(config["save_data"],index=False)
    else:
        raise ValueError("'save_data' need to specify a path")
    
    return df


def run_loading(args):
    """Loads config and executes load data set
    Args:
        args: From argparse, should contain args.config and args.save if otherwise specified
        args.config (str): Path to yaml file with load_data as a top level key containing relevant
        configurations
        args.save (str): Optional. If given, resulting dataframe will be saved to this location.
   
   Returns: None
    """
    with open(args.config, "r") as f:
        config = yaml.load(f)

    df = load_data(**config["load_data"])

    if args.save is not None:
        df.to_csv(args.save, index=False)
    

