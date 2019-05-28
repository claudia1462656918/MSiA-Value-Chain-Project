"""
This module includes unit tests for two important functions in data_process_helper.py.
"""

import pandas as pd
import numpy as np
import generate_features as helper



def test_drop_irrelevant_predictors():
    """Testing drop_irrelevant_predictors function."""

    # data frame inputs
    df_inputs = {
        "Unnamed: 0":[0,1,2],
        "duration": [10,20,30],
        "age": [33,26,45]
    }
    df_testing = pd.DataFrame(data=df_inputs)

    # actual output
    actual = helper.drop_irrelevant_predictors(df_testing)

    # expected output
    df_expected = {
        "age": [33,26,45],
    }
    expected = pd.DataFrame(data=df_expected)

    try:
        # check type
        assert isinstance(expected, pd.DataFrame)

        # check expected output
        assert actual.equals(expected)
        print('Test for drop_irrelevant_predictors function PASSED!')
    except:
        print('Test for drop_irrelevant_predictors function FAILED!')


def test_additional_processing():
    """Testing additional_processing function."""

    # data frame inputs
    df_inputs = {
        "job": ['admin.','blue-collar','entrepreneur'],
        "age": [33,26,45]
    }
    df_testing = pd.DataFrame(data=df_inputs)

    # actual output
    actual = helper.additional_processing(df_testing)

    # expected output
    df_expected = {
        "job": ['admin','blue-collar','entrepreneur'],
        "age": [33,26,45]
    }
    expected = pd.DataFrame(data=df_expected)

    try:
        # check type
        assert isinstance(expected, pd.DataFrame)

        # check expected output
        assert actual.equals(expected)
        print('Test for additional_processing function PASSED!')
    except:
        print('Test for additional_processing function FAILED!')

        
        
        
def test_convert_types():
    """Testing convert_types function."""

    # data frame inputs
    df_inputs = {
         "job": ['admin','blue-collar','admin'],
        "contact": ['cellular', 'telephone', 'unknown']
    }
    df_testing = pd.DataFrame(data=df_inputs)

    # actual output
    actual = helper.convert_types(df_testing)

    # expected output
    df_expected = {
        "job": [0,1,0],
        "contact": [0,1,2]
    }
    expected = pd.DataFrame(data=df_expected)

    try:
        # check type
        assert isinstance(expected, pd.DataFrame)
        
        # check expected output
        assert actual.equals(expected)
        print('Test for convert_types function PASSED!')
    except:
        print('Test for convert_types function FAILED!')

        
def main():
    """Main function"""

    test_drop_irrelevant_predictors()
    test_additional_processing()
    test_convert_types()

if __name__ == "__main__":
    main()