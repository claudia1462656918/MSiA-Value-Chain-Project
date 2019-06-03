import logging
import argparse
import yaml
import os
import subprocess
import re
import datetime

import sklearn
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score,f1_score

logger = logging.getLogger(__name__)


def evaluate_model(df_y_true, y_predicted, save_evaluation=None, **kwargs):
    """Evaluate the performance of the model   
    Args:
        df_y_true (:py:class:`pandas.DataFrame`): Dataframe containing true y class label
        y_predicted (:py:class:`pandas.DataFrame`): Dataframe containing predicted probability and class label
    Returns: 
        confusion_df (:py:class:`pandas.DataFrame`): Dataframe reporting confusion matrix
    """
    # get predicted probability of buying the financial product of the bank 
    y_pred_prob = y_predicted.iloc[:,0]

    # calculate auc and accuracy and f1_score if specified
    if "accuracy" in kwargs["metrics"]:
        # second parameter is the predicted label of class y 
        accuracy = accuracy_score(df_y_true, y_predicted.iloc[:,1])
    if "auc" in kwargs["metrics"]:
        # second parameter is the predicted prob of target y 
        auc = roc_auc_score(df_y_true, y_predicted.iloc[:,0])
    if "f1_score" in kwargs["metrics"]:
        # second parameter is the predicted label of class y 
        f1 = f1_score(df_y_true, y_predicted.iloc[:,1])
    
       
    # get the confusion matrix

    # second parameter is the predicted label of class y 
    confusion = confusion_matrix(df_y_true, y_predicted.iloc[:,1])
    confusion_df = pd.DataFrame(confusion,index=['Actual: Negative','Actual: Positive'])
    confusion_df.columns = ['Predicted: Negative', 'Predicted: Positive']
    print(confusion_df)
    print('\n')
    # calculate other metric 
    metric = pd.DataFrame({'auc':[auc],'accuracy': [accuracy], 'f1':[f1]})
    # verify 
    print(metric)
    return metric


def run_evaluation(args):
    """Loads config and executes calculation for accuracy and auc scores 
    Args:
        args: From argparse, should contain args.config and optionally, args.save
            args.config (str): Path to yaml file with evaluate_model as a top level key containing relevant configurations
            args.input (str): Optional. If given, resulting dataframe will be used in score calculations
            args.output (str): Optional. If given, resulting dataframe will be saved to this location.
    Returns: None
    """

    # load the section of evaulation in the config file 
    with open(args.config, "r") as f:
        config = yaml.load(f)

    if "score_model" in config and "save_scores" in config["score_model"]:
        logger.info("loading the predicted scores")
        score_df = pd.read_csv(config["score_model"]["save_scores"])
        
    else:
        raise ValueError("we cannot find the score_model' in config file")
        print('please give a csv path to read or fix your config file corresponding section')


    # read the input file here 
    if args.input is not None:
        df = pd.read_csv(args.input)
    elif "train_model" in config and "split_data" in config["train_model"] and "save_split_prefix" in config["train_model"]["split_data"]:
        logger.info("load the target varaible y now")
        df = pd.read_csv(config["train_model"]["split_data"]["save_split_prefix"]+ "-test-targets.csv")
        print('read in y in test data')
    else:
        raise ValueError("There is no path to access the input data given in the --input or config file of train_model section")
        print('please give a path to load csv')
    
    # generate the result metric calling the function 
    confusion_df = evaluate_model(df, score_df, **config["evaluate_model"])
    
    if args.output is not None:
        confusion_df.to_csv(args.output)
    elif "evaluate_model" in config and "save_evaluation" in config["evaluate_model"]:
        confusion_df.to_csv(config["evaluate_model"]["save_evaluation"])
    else:
        raise ValueError("We cannot find the csv path for data, neither in --output nor "
                         "'evaluate_model' section in config file")
        print('please give a correct input path or fix corresponding section in config file')

        

