import pandas as pd
from sklearn.utils import resample
import numpy as np
from src.load_data import load_data

def drop_irrelevant_predictors(df,del_column):
    """
    Function that drops bad predictors.
    This function drops attribute with meaningless values such as index number,
    and attribute that highly affects the output target (e.g., if duration=0 then y='no'). 
    Yet, the duration is not known before a call is performed. Also, after the end of 
    the call y is obviously known. 
  
    Args:
        df (pandas.DataFrame object): data frame that holds all data
    Returns:
        df (pandas.DataFrame object): data frame with bad predictors removed
    """
    
    df.drop(del_column, inplace=True, axis=1)
    
    return df


def additional_processing(df,column):
    """
    Function that clean the column job, remove the trailing dot 
    
    Args:
        df (pandas.DataFrame object): data frame with irrevant predictors removed
    Returns:
        df (pandas.DataFrame object): data frame with trailing dots removed in cells 
    """
    
    df[column] = df[column].replace('\.','', regex=True)
    return df

def reduce_no_of_class(df):
    """
    Function that reduce the number of classes for the month and bins numerical age variables into high, medium, and low
    
    Args:
        df (pandas.DataFrame object): data frame with irrevant predictors removed
    Returns:
        df (pandas.DataFrame object): data frame that has variable of month with 3 classes and age with 3 classes 
    """
    
    j = 0    
    for i in df['month']:
        if (i in ('may','jun','jul','aug')):
            df.iloc[j,10] = 'high'
        elif (i in ('feb','apr','nov')):
            df.iloc[j,10] = 'medium'
        else:
            df.iloc[j,10] = 'low'
        j = j+1
    
    c1 = df['age'].quantile(1/3)
    c2 = df['age'].quantile(2/3)

    j = 0
    for i in df['age']:
        if (i < c1):
            df.iloc[j,0] = 'low'
        elif (c1 <=i<=c2 ):
            df.iloc[j,0] = 'medium'
        else:
            df.iloc[j,0] = 'high'
        j = j+1
    return df


    
    
    
def balance_class(df):
    """
    Function that balances y.
    The response variable y is imbalanced with no:yes roughly equal to 9:1.
    This function down-samples the majority class by randomly removing observations
    from the majority class to prevent its signal from dominating the learning algorithm.
    Args:
        df (pandas.DataFrame object): data frame
    Returns:
        df_downsampled (pandas.DataFrame object): downsampled data frame
    """
    # separate observations from each class into different DataFrames
    # resample the majority class without replacement,
    # setting the number of samples to match that of the minority class
    # combine the down-sampled majority class DataFrame with the original minority class DataFrame.

    # Separate majority and minority classes
    df_majority = df[df.y == 'no']
    df_minority = df[df.y == 'no']

    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=521,  # to match minority class
                                       random_state=123)  # reproducible results

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    return df_downsampled


def convert_types(df):
    """
    Function that converts categorical attributes to numerical factors
    that can be used in future model development.
    
    Args:
        df (pandas.DataFrame object): cleaned and undersampled data
    Returns:
        df (pandas.DataFrame object): data frame with only numbers
    """
    
    for col_name in list(df):
        if(df[col_name].dtype == 'object'):
            df[col_name]= df[col_name].astype('category')
            df[col_name] = pd.factorize(df[col_name])[0]
    return df


def run_data_process(args):
    if args.input is not None:
        df = pd.read_csv(args.input)
    elif "load_data" in config:
        df = load_data(config["load_data"])
    else:
        raise ValueError("Path to CSV for input data must be provided through --csv or "
                         "'load_data' configuration must exist in config file")
        
        
    df = drop_irrelevant_predictors(**config["drop_irrelevant_predictors"]) 
    df = additional_processing(**config["additional_processing"])
    df = balance_class(df)
    df = reduce_no_of_class(df)
    df = convert_types(df)
    
    if args.output is not None:
        df.to_csv(args.output, index=False)
        logger.info("Features saved to %s", args.output)

    return df








