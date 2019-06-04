
"""
This module includes unit tests for two important functions in data_process_helper.py.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pytest
import sklearn
import src.generate_features as gf
import src.train_model as tm
import src.score_model as sm 
import src.evaluate_model as em




def test_get_target():
    """ testing whether we get the value of target """

    # data frame input 
    df_inputs ={'x':[1,2,3],
                'y':[4,5,6]
                }
    df_testing = pd.DataFrame(data=df_inputs)

    # actual output
    actual = gf.get_target(df_testing,'y')

    # expected output
    df_expected = np.array([4,5,6])

    expected = df_expected

    try:
        # check type
        assert isinstance(expected, np.ndarray)

        # check expected output
        assert np.array_equal(actual,expected)
        print('Test for get_target function PASSED!')
    except:
        print('Test for get_target function FAILED!')



def test_additional_processing():
    """Testing additional_processing function."""

    # data frame inputs
    df_inputs = {
        "job": ['admin.','blue-collar','entrepreneur'],
        "age": [33,26,45]
    }
    df_testing = pd.DataFrame(data=df_inputs)

    # actual output
    actual = gf.additional_processing(df_testing,'job')

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

        

def test_reduce_no_of_class():
    """testing whether we balance the data using the downsampling method"""
    # data frame input 
    df_inputs ={
                'age':[1,2,3,4,5,6,7],
                'month':['jan','feb','mar','apr','may','jun','july'],
                'month1':['jan','feb','mar','apr','may','jun','july'],
                'month2':['jan','feb','mar','apr','may','jun','july'],
                'month3':['jan','feb','mar','apr','may','jun','july'],
                'month4':['jan','feb','mar','apr','may','jun','july'],
                'month5':['jan','feb','mar','apr','may','jun','july'],
                'month6':['jan','feb','mar','apr','may','jun','july'],
                'month7':['jan','feb','mar','apr','may','jun','july'],
                'month8':['jan','feb','mar','apr','may','jun','july'],
                'month_new':['jan','feb','mar','apr','may','jun','july']
                }
    df_testing = pd.DataFrame(data=df_inputs)
    
    # actual output
    actual = gf.reduce_no_of_class(df_testing)
    
    # expected output
    df_expected = {
        'age':['low','low','medium','medium','high','high','high'],
        'month':['jan','feb','mar','apr','may','jun','july'],
        'month1':['jan','feb','mar','apr','may','jun','july'],
        'month2':['jan','feb','mar','apr','may','jun','july'],
        'month3':['jan','feb','mar','apr','may','jun','july'],
        'month4':['jan','feb','mar','apr','may','jun','july'],
        'month5':['jan','feb','mar','apr','may','jun','july'],
        'month6':['jan','feb','mar','apr','may','jun','july'],
        'month7':['jan','feb','mar','apr','may','jun','july'],
        'month8':['jan','feb','mar','apr','may','jun','july'],
        'month_new':['low','medium','low','medium','high','high','low']
    }

    expected = pd.DataFrame(data=df_expected)
    
    try:
        # check type
        assert isinstance(expected, pd.DataFrame)

        # check expected output
        assert actual.equals(expected)
        print('Test for reduce_no_of_class function PASSED!')
    except:
        print('Test for reduce_no_of_class function FAILED!')
        

        
def test_convert_types():
    """Testing convert_types function."""

    # data frame inputs
    df_inputs = {
         "job": ['admin','blue-collar','admin'],
        "contact": ['cellular', 'telephone', 'unknown']
    }
    df_testing = pd.DataFrame(data=df_inputs)

    # actual output
    actual = gf.convert_types(df_testing)

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



def test_choose_features():
    """Testing choose_features function."""

    # data frame inputs
    df_inputs = {
         "job": ['admin','blue-collar','admin'],
        "contact": ['cellular', 'telephone', 'unknown'],
        "y":[1,2,3]
    }
    df_testing = pd.DataFrame(data=df_inputs)

    # actual output
    features_to_use = ['job']
    target = 'y'

    actual = gf.choose_features(df_testing, features_to_use,target )

    # expected output
    df_expected = {
        "job": ['admin','blue-collar','admin'],
        "y":[1,2,3]
    }
    expected = pd.DataFrame(data=df_expected)

    try:
        # check type
        assert isinstance(expected, pd.DataFrame)
        
        # check expected output
        assert actual.equals(expected)
        print('Test for choose_features function PASSED!')
    except:
        print('Test for choose_features function FAILED!')




def test_split_data():
    """a fucntion to test split_data()"""
    
    # data frame inputs
    X = pd.DataFrame({
        "x": [1,2,3,4,5,6,7,8,9,10]})

    y = pd.DataFrame({
        'y':[11,12,13,14,15,16,17,18,19,20]
        
    })
    kwargs = {'train_size':0.7, 'test_size':0.3, 'random_state': 42}
    X,y = tm.split_data(X, y,**kwargs )

    # expected output
    x_train = pd.DataFrame({
        "x": [1,8,3,10,5,4,7]})
    x_test =  pd.DataFrame({
            "x": [9,2,6]})
    X_expected = {'train':x_train,'test':x_test}
    X_expected['train']
    X['train'] = X['train'].reset_index(drop=True)
    
    try:
        # check type
        assert isinstance(X_expected['train'], pd.DataFrame)
        # check expected output
        assert X_expected['train'].equals(X['train'])
        print('Test for split_data function PASSED!')
    except:
        print('Test for split_data function FAILED!')





def test_evaluate_model():
    """Testing evaluate_model function."""

    # data frame inputs
    df_y_true = pd.DataFrame({
     "y": [1,0,0]})

    y_predicted = pd.DataFrame({
        'prob':[0.7,0.6,0.4],
        'label':[1,1,1]
    })
    kwargs = {'metrics': ['auc','accuracy', 'f1_score']}
    actual = em.evaluate_model(df_y_true, y_predicted,**kwargs )

    # expected output
    df_expected = {
        "auc": [1.0],
        "accuracy":[1/3],
        'f1':[0.5]
    }
    expected = pd.DataFrame(data=df_expected)
    
    try:
        # check type
        assert isinstance(expected, pd.DataFrame)
        # check expected output
        assert actual.equals(expected)
        print('Test for evaluate_model function PASSED!')
    except:
        print('Test for evaluate_model function FAILED!')





def test_train_model():
    """test if output of train_model function is an instance of xgboost model class"""
    
    df = pd.read_csv('data/bank_processed.csv')
    method = 'xgboost'

    kwargs = {'params':{"max_depth":3, 'n_estimators': 300,'learning_rate': 0.05},
              'split_data': {'train_size':0.7, 'test_size':0.3, 'random_state': 42},
              "get_target":{'target':'y'}, 
              'choose_features':{'features_to_use': 
                                 ['age', 'job', 'marital', 'education', 
                                  'default', 'balance', 'housing','loan', 
                                  'contact', 'day', 'month', 'campaign', 
                                  'pdays', 'previous','poutcome']}
             }

    model = tm.train_model(df, method, **kwargs)
 
    try:
        # check type
        assert isinstance(model, xgb.XGBClassifier)
        
        print('Test for train_model function PASSED!')
    except:
        print('Test for train_model function FAILED!')




def test_train_model_input():
    """Test whether train_model works as expected as only accept numberic/bool input data"""
    with pytest.raises(ValueError) as message:

        # load sample test data
        df = pd.read_csv('data/bank_processed.csv')
        df['age'] = df['age'].astype(str)
        print(type(df['age']))
        
        method = 'xgboost'

        kwargs = {'params':{"max_depth":3, 'n_estimators': 300,'learning_rate': 0.05},
                  'split_data': {'train_size':0.7, 'test_size':0.3, 'random_state': 42},
                  "get_target":{'target':'y'}, 
                  'choose_features':{'features_to_use': 
                                     ['age', 'job', 'marital', 'education', 
                                      'default', 'balance', 'housing','loan', 
                                      'contact', 'day', 'month', 'campaign', 
                                      'pdays', 'previous','poutcome']}
                 }

        model = tm.train_model(df, method, **kwargs)



    # raise AssertionError if error message is not as expected
    # remove trailing white space and space in the message 
    assert str(message.value).replace(" ", "").replace('\n','') == 'DataFrame.dtypesfordatamustbeint,floatorbool.Didnotexpectthedatatypesinfieldsage'




def test_train_model_output():
    """Test whether the train_model function gives the expected output as we do using xgb.classifier normally"""
    
    # read the data
    df = pd.read_csv('data/bank_processed.csv')
    methods = dict(xgboost=xgb.XGBClassifier)
    
    # specific parameters used in config 
    max_depth = 3
    n_estimators = 300
    learning_rate = 0.05
    
    method = 'xgboost'
    
    kwargs = {'params':{"max_depth":3, 'n_estimators': 300,'learning_rate': 0.05},
              'split_data': {'train_size':0.7, 'test_size':0.3, 'random_state': 77},
              "get_target":{'target':'y'}, 
              'choose_features':{'features_to_use': 
                                 ['age', 'job', 'marital', 'education', 
                                  'default', 'balance', 'housing','loan', 
                                  'contact', 'day', 'month', 'campaign', 
                                  'pdays', 'previous','poutcome']}
             }
    xgb_expected = tm.train_model(df, method, **kwargs)

    # features used and target variable
    X_df = df[['age', 'job', 'marital', 'education', 
              'default', 'balance', 'housing','loan', 
              'contact', 'day', 'month', 'campaign', 
              'pdays', 'previous','poutcome']]
    y_df = df['y']
    
    #split the data to train and test set  
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_df, y_df, 
        train_size=0.7, test_size=0.3, random_state=77)
    
    # train a xgb model with given parameters 
    xgb_actual = xgb.XGBClassifier(objective='binary:logistic', n_estimators=n_estimators,
                                           max_depth=max_depth,learning_rate=learning_rate)  
    xgb_actual.fit(X_train,y_train)

    # check if the result model is as expected using the train_model function 
    assert str(xgb_actual.get_xgb_params) == str(xgb_expected.get_xgb_params)
    # check if the xbg model has the feature_importances_ method 
    assert xgb_expected.feature_importances_ is not np.nan



def test_score_model():
    """test the function of score_model whether it gives correct probability between [0,1]"""
    
    df = pd.read_csv('data/bank_processed.csv')
    path_to_tmo = 'models/bank-prediction.pkl'
    cutoff = 0.5
    
    kwargs = {"choose_features": {'features_to_use': 
                                  ['age', 'job', 'marital', 'education', 
                                   'default', 'balance', 'housing','loan', 
                                   'contact', 'day', 'month', 'campaign', 
                                   'pdays', 'previous','poutcome']}}
    actual = sm.score_model(df, path_to_tmo, cutoff, save_scores=None, **kwargs)
    
    n1 = (sum(actual.pred_prob.between(0,1,inclusive=True)))
    n2 = (actual.shape[0])
    try:
        # check type
        assert isinstance(actual, pd.DataFrame)
        # check whether all data probability range is [0,1]
        assert n1==n2
        print('Test for split_data function PASSED!')
    except:
        print('Test for split_data function FAILED!')
    


        
