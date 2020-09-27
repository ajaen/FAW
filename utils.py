import pandas as pd
import numpy as np
import netCDF4
import datetime
import os
from bisect import bisect_left
import random
import xgboost as xgb
import pickle


def drop_constantcolumns(pddf):
    '''This function takes a pandas dataframe as input and returns the same dataframe without the constant and
    without the all NaN columns'''
    returndf = pddf.loc[:, (pddf != pddf.iloc[0]).any()].copy()
    returndf.dropna(axis=1, how='all', inplace=True)
    return(returndf)


def take_closest(myList, myNumber):
    """
    https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before


def catencoder_dict_generator(pddf, threshold=0.02):
    '''This function takes one pandas dataframe (the one we want to encode later) and a threshold between
    0 and 0.5
    The output will be a dictionary indicating: {column_name_string: list of values to be one-hot-encoded, ...}
    The values to be one-hot-encoded will be the ones that represent certain percentage of the total number
    of values (if threshold is 0.02 this percentage is 2%)
    '''
    total = pddf.shape[0]
    encoding_dict = {}
    for catcol in pddf.columns[pddf.dtypes == object]:
        ls = list(pddf[catcol].value_counts().index[pddf[catcol].value_counts() / total > threshold])
        if len(ls) > 0:
            if (len(ls) > 1) & (len(ls) < pddf[catcol].unique().size - 1):
                ls.append('rest')
            encoding_dict[catcol] = ls
    return(encoding_dict)



def catencoder(pddf, enc_dict, idcol='user_id'):
    '''Returns a dataframe with the categorical features one-hot-encoded following the given dictionary
    which contains the columns+values to be binarily encoded'''
    for key in enc_dict:
        for value in enc_dict[key]:
            if value == 'rest':
                pddf[key + '_rest'] = pddf.iloc[:, -len(enc_dict[key]):].sum(axis=1) # Value rest would only be in the last position of the enc_dict[key] list
            else:
                pddf[key + '_' + value] = (pddf[key] == value).astype(int)
    pddf.drop(pddf.columns[pddf.dtypes == object].drop(idcol), axis=1, inplace=True)
    return(pddf)


def evaluate_model(tuple_params, tr, metric, maximize_metric=None, objective_type=None, tr_fcolumnvalue=0, eval_fcolumnvalue=1):
    '''Given a tuple indicating (max_depth, min_child_weight, subsample, colsample) this function
    returns the AUC of an xgboost trained in our train set (tr dataframe where folds_column is equal to 0)
    and evaluated in out validation set (tr dataframe where folds_column is equal to 1) using the given
    parameters.
    This function assumes the existence of certain fields in the dataframe tr
    
    metric:
    Metric to be optimize when choosing the XGBoost parameters:
    (read https://github.com/dmlc/xgboost/blob/master/doc/parameter.md to see the available metrics)
    Examples:
    Categoric target with more than two labels: 'mlogloss'
    Binary target: 'logloss', 'auc'
    Numeric target: 'rmse', 'mae'
    
    maximize_metric:
    Specify if we are using a metric which should be maximized (True) o minimized (False). (only
    when the metric used is different of the ones considered in the previous examples) AUTO otherwise
    
    objective_type:
    Metric to be used internally by XGBoost to select the proper splits in each tree. We'll normally use:
    'multi:softprob' for categoric targets with more than two labels
    'binary:logistic' for binary targets
    'reg:linear' if we are predicting a non-categoric target (regression)
    to see other possible metrics, read http://xgboost.readthedocs.io/en/latest/parameter.html
    We'll AUTOmatize this assignment based on the metric parameter value (specify when necessary)
    '''

    if metric in ('mlogloss', 'logloss', 'rmse', 'mae'):
        maximize_metric = False
    elif metric == 'auc':
        maximize_metric = True

    if metric == 'mlogloss':
        objective_type = 'multi:softprob'
    elif metric in ('logloss', 'auc'):
        objective_type = 'binary:logistic'
    elif metric in ('rmse', 'mae'):
        objective_type = 'reg:linear'

    xgtrain = xgb.DMatrix(tr.drop(['user_id', 'folds_column', 'target'], axis=1),
                          label = tr.target.astype(int),
                          # missing = NAs_value,
                          feature_names = np.delete(tr.columns.values,
                                                    np.where(np.isin(tr.columns.values,
                                                                     ['user_id', 'folds_column', 'target'])))
                         )

    xgb_param = {}

    xgb_param['gamma'] = 0
    xgb_param['learning_rate'] = 0.1 # Dejamos el learning rate fijo.
    xgb_param['n_estimators'] = 5000 # Máximo número de árboles para un sólo modelo
    xgb_param['booster'] = 'gbtree'
    xgb_param['objective'] = objective_type
    xgb_param["eval_metric"] = metric
    # xgb_param['num_class'] = len(np.unique(xgtrain.get_label()))
    xgb_param['silent'] = 1
    xgb_param['seed'] = 49 # Esta semilla se usa para el entrenamiento del modelo
    xgb_param['base_score'] = 0.5

    xgb_param['max_depth'] = tuple_params[0]
    xgb_param['min_child_weight'] = tuple_params[1]
    xgb_param['subsample'] = tuple_params[2]
    xgb_param['colsample_bytree'] = tuple_params[3]

    XGB_model = xgb.train(xgb_param,
                          xgtrain.slice(np.where(tr.folds_column == tr_fcolumnvalue)[0]),
                          num_boost_round=5000,
                          early_stopping_rounds=10,
                          evals = [#(xgtrain.slice(np.where(tr.folds_column == 2)[0]),'test'),
                                   (xgtrain.slice(np.where(tr.folds_column == eval_fcolumnvalue)[0]),'eval')],
                          verbose_eval = False,
                          maximize = maximize_metric)
#####################################################################################################################
    # (XGB_model.best_score, XGB_model.best_iteration)
#####################################################################################################################
    return((XGB_model.best_score, XGB_model.best_iteration))



def evaluate_grid(gridsearch_params, tr, metric, maximize_metric=None, objective_type=None, tr_fcolumnvalue=0, eval_fcolumnvalue=1):
    '''This function returns the tuple of parameters whose correspondent xgboost trained in the instances
    from tr whose folds_columns value is tr_fcolumnvalue, performs the best in the instances from tr whose
    folds_columns value is eval_fcolumnvalue
    '''

    if metric in ('mlogloss', 'logloss', 'rmse', 'mae'):
        maximize_metric = False
    elif metric == 'auc':
        maximize_metric = True

    if metric == 'mlogloss':
        objective_type = 'multi:softprob'
    elif metric in ('logloss', 'auc'):
        objective_type = 'binary:logistic'
    elif metric in ('rmse', 'mae'):
        objective_type = 'reg:linear'

    grid_aucs = np.array(list(map(lambda mytuple: evaluate_model(mytuple,
                                                                 tr,
                                                                 metric,
                                                                 maximize_metric,
                                                                 objective_type,
                                                                 tr_fcolumnvalue=0,
                                                                 eval_fcolumnvalue=1),
                              gridsearch_params)))
    return(list(gridsearch_params[grid_aucs[:, 0].argmax()]) + [grid_aucs[grid_aucs[:, 0].argmax(), 1],] if maximize_metric else list(gridsearch_params[grid_aucs[:, 0].argmin()]) + [grid_aucs[grid_aucs[:, 0].argmin(), 1],])

def evaluate_grid2(gridsearch_params, tr, metric, maximize_metric=None, objective_type=None, tr_fcolumnvalue=0, eval_fcolumnvalue=1):
    '''This function returns the tuple of parameters whose correspondent xgboost trained in the instances
    from tr whose folds_columns value is tr_fcolumnvalue, performs the best in the instances from tr whose
    folds_columns value is eval_fcolumnvalue
    '''

    if metric in ('mlogloss', 'logloss', 'rmse', 'mae'):
        maximize_metric = False
    elif metric == 'auc':
        maximize_metric = True

    if metric == 'mlogloss':
        objective_type = 'multi:softprob'
    elif metric in ('logloss', 'auc'):
        objective_type = 'binary:logistic'
    elif metric in ('rmse', 'mae'):
        objective_type = 'reg:linear'

    grid_aucs = np.array(list(map(lambda mytuple: evaluate_model(mytuple,
                                                                 tr,
                                                                 metric,
                                                                 maximize_metric,
                                                                 objective_type,
                                                                 tr_fcolumnvalue=0,
                                                                 eval_fcolumnvalue=1),
                              gridsearch_params)))
    return(grid_aucs)

def auxmygrid(gridspace_list):
    return(list(map(random.choice, gridspace_list)))

def mygrid(gridspace_list, n_iter=100):
    return(list(map(auxmygrid, [gridspace_list for i in range(n_iter)])))
