import logging
import argparse
import yaml
import os
import subprocess
import re
import datetime

import pickle

import sklearn
import pandas as pd
import numpy as np

from load_data import load_data
from generate_features import choose_features, get_target
import xgboost as xgb

logger = logging.getLogger(__name__)




def score_model(df, path_to_tmo, cutoff, save_scores=None, **kwargs):
    """Given the prediction about whether customers buy the bank product  for the test set.
    Args:
        df (:py:class:`pandas.DataFrame`): Dataframe with the bank data to give prediction.
        path_to_tmo (str): trained model path
        cutoff (int): the cutoff point where customer with probability above it will classifu as buying the financial product and vice versus
        save_scores (str): path where the result of the prediction values are saved 
        
    Returns:
        y_predicted (:py:class:`pandas.DataFrame`): 2 column dataframe with predicted probability and class.
    
    """
    # load the saved xgboost model 
    with open(path_to_tmo, "rb") as f:
        model = pickle.load(f)
    # only choose the features specified in the config section
    if "choose_features" in kwargs:
        X = choose_features(df, **kwargs["choose_features"])
    else:
        X = df

    # features_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing','loan', 'contact', 'day', 'month', 'campaign', 'pdays', 'previous','poutcome']
    # new_df = pd.DataFrame(X, columns = features_columns)
    #result = loaded_model.predict(new_df)
    # just get the probability of customers buying the product, initially 2 columns, one is for not buy, one is for buy 
    y_prob_yes = model.predict_proba(X)[:,1]

    #print(model.predict_proba(X))
    y_pred = pd.DataFrame(y_prob_yes)
    # name the probability column of the class who does want to buy the product  
    y_pred.columns = ['pred_prob']
    # add a new column that convert the probability to the class label using the helper function 
    y_pred['pred'] = [1 if i>cutoff else 0 for i in y_prob_yes]

    #print(len(y_pred.columns))
    
    if len(y_pred.columns) == 2:
            logger.info("The following columns are included in scores: %s", ",".join(y_pred.columns))

    # save the results to the given path 
    if save_scores is not None:
        y_pred.to_csv(save_scores, index=False)

    return y_pred


def run_scoring(args):
    """Loads config and executes class prediciton both for class and probability 
    Args:
        args: From argparse, should contain args.config and optionally, args.save
            args.config (str): Path to yaml file with score_model as a top level key containing relevant configurations
            args.input (str): Optional. If given, resulting dataframe will be used in outcome prediction 
            args.output (str): Optional. If given, resulting dataframe will be saved to this location.
    Returns: None
    """
    
    with open(args.config, "r") as f:
        config = yaml.load(f)

    if args.input is not None:
        x_df = pd.read_csv(args.input)
    # check whether the given input exist in the config files so need to check the train_model section
    elif "train_model" in config and "split_data" in config["train_model"] and "save_split_prefix" in config["train_model"]["split_data"]:
        # read in x_test, prepared for later prediction of y_pred
        x_df = pd.read_csv(config["train_model"]["split_data"]["save_split_prefix"]+ "-test-features.csv")
    else:
        raise ValueError("Path to CSV for input data must be provided through --input or "
                         "'load_data' configuration must exist in config file")
    # Get predicted scores of the test set.
    score_result = score_model(x_df, **config["score_model"])

    if args.output is not None:
        pd.DataFrame(score_result).to_csv(args.output, index=False)

        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Score model")
    parser.add_argument('--config', '-c', help='path to yaml file with configurations')
    parser.add_argument('--input', '-i', default=None, help="Path to CSV for input to model scoring")
    parser.add_argument('--output', '-o', default=None, help='Path to where the scores should be saved to (optional)')

    args = parser.parse_args()

    run_scoring(args)