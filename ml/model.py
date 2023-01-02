"""
Author: Ali Kilinc
Date: 31.12.2022

This file is holding necessary functions to train ML models, evaluate them and create slicewise performance evalutations. 

LightGBM model was used to train the data, also a GridSearchCV schema with a preprocessor pipeline was applied inside. 

It selects the best model among all of the options, saves this best model as best estimator, and saves it to use on Validation Set put aside at the beginning of the model training process. 

Since GridSearchCV trains best model with all data when Refit=True is set, this model also does this. By using this pipeline object, only required thing to make predictions on a test set is .predict function. 

By using .predict function, we are automatically applying preprocessing step fittet for Training data. It applies same preprocessing to Test set, then make predictions on this one. 
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
from datetime import datetime
import timeit
import numpy as np
import logging

logging.basicConfig(filename='model_logging.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

# Optional: implement hyperparameter tuning.
def train_model(cat_cols, num_cols, X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Used GridSearchCV and LightGBM model for training

    Inputs
    ------
    cat_cols: list
        List of categorical columns
    num_cols: List
        List of numerical columns
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    est
        Trained machine learning model, best one among all parameter grid
    """
    
    parameters = {
            'est__max_depth':[5, 9, 18],
            'est__learning_rate': [0.01, 0.1],
            'est__n_estimators': [10, 50, 100],
            'est__num_leaves': np.linspace(11,51,3,endpoint = True, dtype = int)
            }
    
    start_time = timeit.default_timer()
        
    scorer = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall']
        
    preprocessor = ColumnTransformer(transformers = [('num', MinMaxScaler(feature_range = (0,1)), num_cols),
                                                         ('cat', OneHotEncoder(sparse = True, handle_unknown = 'ignore'), cat_cols)])
        
    all_pipe = Pipeline(steps = [('prep', preprocessor),('est', LGBMClassifier(random_state = 23))])
        
    search_space = parameters
        
    #njobs = -1 uses 2 cores (all available) and 4 threads; consumes 100% of CPU. But it is faster absolutely. Njobs = 1 uses 1 core only, slower but consumes less CPU
    grid_search = GridSearchCV(all_pipe, search_space, cv=5, verbose=0, refit = 'roc_auc', scoring = scorer, return_train_score = True, n_jobs = 4)
    #scoring option is for defining multiple scorers, otherwise null is OK
    #refit must be chosen if multi scorers is selected. But best_score_, best_params_ etc will give only the result of that metric, cant get multi scores
    #here REFIT option tells you which metric do you want to consider for best_params_ calculation(according to which metric)
        
    #REFIT = True for gridsearchcv means, at the end train with full data with best estimator, to get final model, 
    #grid_search.best_estimator_ can be used as model file for Pickle Dump. It is the best model estimator, 
    # With refit you are training whole data with whole pipeline. So that, you can just use PREDİCT object for this pipe to predict unseen data, without preprocessing seperately
    grid_search.fit(X_train, y_train)
    
    est = grid_search.best_estimator_
    
    ind = grid_search.best_index_      
        
    logging.info("model = {}".format("LGBClassifier"))
    logging.info("train_roc = {}, test_roc ={}".format(grid_search.cv_results_['mean_train_roc_auc'][ind], grid_search.cv_results_['mean_test_roc_auc'][ind]))
    logging.info("train_accuracy = {}, test_accuracy ={}".format(grid_search.cv_results_['mean_train_accuracy'][ind], grid_search.cv_results_['mean_test_accuracy'][ind]))
        
    logging.info("train_f1 = {}, test_f1 = {}".format(grid_search.cv_results_['mean_train_f1'][ind], grid_search.cv_results_['mean_test_f1'][ind]))
    logging.info("train_precision = {}, test_precision = {}".format(grid_search.cv_results_['mean_train_precision'][ind], grid_search.cv_results_['mean_test_precision'][ind]))
    logging.info("train_recall = {}, test_recall = {}".format(grid_search.cv_results_['mean_train_recall'][ind], grid_search.cv_results_['mean_test_recall'][ind]))
    
    logging.info("avg_fit_time = {}".format(grid_search.cv_results_['mean_fit_time'][ind]))
    #we are getting best_params_ from grid_search object still, it is valid
    logging.info("best_params = {}".format(grid_search.best_params_))
    logging.info("---%0.1f minutes---" %((timeit.default_timer()-start_time)/60))
    logging.info("...{}...".format(start_time))
    logging.info("current_time = {}...".format(datetime.now()))
    logging.info("current_time = {}...".format(datetime.now()))
    logging.info("-----------------------------------")

    return est


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Model object
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    
    preds = model.predict(X)
    return preds

def compute_confusion_matrix(y, preds, labels=None):
    """
    Compute confusion matrix using the predictions and ground thruth provided
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    cm : confusion matrix for the provided prediction set
    """
    cm = confusion_matrix(y, preds)
    return cm


def compute_slices(df, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature
    a slice corresponds to one value option of the categorical feature analyzed
    ------
    df: 
        test dataframe pre-processed with features as column used for slices
    feature:
        feature on which to perform the slices
    y : np.array
        corresponding known labels, binarized.
    preds : np.array
        Predicted labels, binarized
    Returns
    ------
    Dataframe with
        n_samples: integer - number of data samples in the slice
        precision : float
        recall : float
        fbeta : float
    row corresponding to each of the unique values taken by the feature (slice)
    """    
    slice_options = df[feature].unique().tolist()
    perf_df = pd.DataFrame(index=slice_options, 
                            columns=['feature','n_samples','precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask = df[feature]==option

        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        
        perf_df.at[option, 'feature'] = feature
        perf_df.at[option, 'n_samples'] = len(slice_y)
        perf_df.at[option, 'precision'] = precision
        perf_df.at[option, 'recall'] = recall
        perf_df.at[option, 'fbeta'] = fbeta

    # reorder columns in performance dataframe
    perf_df.reset_index(names='feature value', inplace=True)
    colList = list(perf_df.columns)
    colList[0], colList[1] =  colList[1], colList[0]
    perf_df = perf_df[colList]

    return perf_df