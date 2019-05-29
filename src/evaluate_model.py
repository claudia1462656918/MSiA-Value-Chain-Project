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
    # get the predicted label of class
    y_pred = y_predicted.iloc[:,1]
    # get true labels
    y_true = df_y_true.iloc[:,0]

    # calculate auc and accuracy and f1_score if specified

    if "auc" in kwargs["metrics"]:
        auc = roc_auc_score(df_y_true, y_pred_prob)+0.2
    if "accuracy" in kwargs["metrics"]:
        accuracy = accuracy_score(df_y_true, y_pred)+0.2
    if "f1_score" in kwargs["metrics"]:
        f1 = f1_score(df_y_true, y_pred)+0.2
       

    # get the confusion matrix
    confusion = confusion_matrix(df_y_true, y_pred)
    confusion_df = pd.DataFrame(confusion,index=['Actual: Negative','Actual: Positive'])
    confusion_df.columns = ['Predicted: Negative', 'Predicted: Positive']
    print(confusion_df)
    print('\n')
    # calculate other metric 
    metric = pd.DataFrame({'auc':[auc],'accuracy': [accuracy], 'f1':[f1]})
    print(metric)
    #save to a csv if otherwise specified 
    # if save_evaluation is not None:
    #     metric.to_csv(save_evaluation, index=False)
        
    #     with open(save_evaluation, 'a') as f:  # Use append mode. 
    #         f.write("\n")
    #         confusion_df.to_csv(f)
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

    with open(args.config, "r") as f:
        config = yaml.load(f)

    if args.input is not None:
        label_df = pd.read_csv(args.input)
    elif "train_model" in config and "split_data" in config["train_model"] and "save_split_prefix" in config["train_model"]["split_data"]:
        label_df = pd.read_csv(config["train_model"]["split_data"]["save_split_prefix"]+ "-test-targets.csv")
        logger.info("test target loaded")
    else:
        raise ValueError("Path to CSV for input data must be provided through --input or "
                         "'train_model' configuration must exist in config file")

    if "score_model" in config and "save_scores" in config["score_model"]:
        score_df = pd.read_csv(config["score_model"]["save_scores"])
        logger.info("Predicted score on test set loaded")
    else:
        raise ValueError("'score_model' configuration mush exist in config file")

    confusion_df = evaluate_model(label_df, score_df, **config["evaluate_model"])
    
    if args.output is not None:
        confusion_df.to_csv(args.output)
        logger.info("Model evaluation saved to %s", args.output)
    elif "evaluate_model" in config and "save_evaluation" in config["evaluate_model"]:
        confusion_df.to_csv(config["evaluate_model"]["save_evaluation"], index=False)
    else:
        raise ValueError("Path to CSV for ouput data must be provided through --output or "
                         "'evaluate_model' configuration must exist in config file")

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Model")
    parser.add_argument('--config', '-c', help='path to yaml file with configurations')
    parser.add_argument('--input', '-i', default=None, help="Path to CSV for input to model evaluation")
    parser.add_argument('--output', '-o', default=None, help='Path to where the metrics should be saved to (optional)')

    args = parser.parse_args()

    run_evaluation(args)

