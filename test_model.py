"""
Author: Ali Kilinc
Date: 01.01.2023

Using pytest, runnning unit tests on the module model.py

"""

import pytest, os, logging, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from ml.model import inference, compute_model_metrics, compute_confusion_matrix
from ml.data import process_data


@pytest.fixture(scope="module")
def data():
    # code to load in the data.
    csv_path = "/mnt/c/Users/tutkukilinc/nd0821-c3-starter-code/starter/data/census_modified.csv"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def path():
    return "/mnt/c/Users/tutkukilinc/nd0821-c3-starter-code/starter/data/census_modified.csv"
	
@pytest.fixture(scope="module")
def model_path():
    return "/mnt/c/Users/tutkukilinc/nd0821-c3-starter-code/starter/model/trained_model.pkl"


@pytest.fixture(scope="module")
def features():
    """
    Fixture - will return the categorical features as argument
    """
	final_list = []
    cat_features = ['workclass',
					'education',
					'marital-status',
					'occupation',
					'relationship',
					'race',
					'sex',
					'native-country']
	
	target = ['salary']
	num_features = [col for col in data.columns if col not in cat_features + target]
	
	return cat_features, num_features, target


@pytest.fixture(scope="module")
def train_dataset(data, features):
    """
    Fixture - returns cleaned train dataset to be used for model testing
    """
	
	cat_features, num_features, target = features
	
    train, val = train_test_split(data, 
                                test_size=0.20, 
                                random_state=23, 
                                stratify=data['salary']
                                )
								
	X_train, y_train, cat_cols, num_cols, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True)	
	
    return X_train, y_train, val


"""
Test methods
"""
def test_import_data(path):
    """
    Test presence and shape of dataset file
    """
    try:
        data = pd.read_csv(path)

    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    # Check the data shape, to see if there is data
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing import_data: The file doesn't appear to have rows and columns")
        raise err



def test_is_model(model_path):
    """
    Check saved model is present inside the model directory; if not saved anything pass
    """
    savepath = model_path
    if os.path.isfile(savepath):
        try:
            _ = pickle.load(open(savepath, 'rb'))
        except Exception as err:
            logging.error(
            "Testing saved model: Saved model does not appear to be valid")
            raise err
    else:
        pass


def test_inference(train_dataset, model_path):
    """
    Check inference function
    """
    X_train, y_train, val = train_dataset

    savepath = model_path
    if os.path.isfile(savepath):
        model = pickle.load(open(savepath, 'rb'))

        try:
            preds = inference(model, X_train)
        except Exception as err:
            logging.error(
            "Inference cannot be performed on saved model and train data")
            raise err
    else:
        pass

def test_val_dataset(train_dataset):
	"""
	Tests if validation set is created correctly
	"""
	
	X_train, y_train, val = train_dataset
	
	# Check the val shape, to see if there is data
    try:
        assert val.shape[0] > 0
        assert val.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing validation split: The validation dataset doesn't appear to have rows and columns")
        raise err
	

def test_compute_model_metrics(train_dataset, model_path):
    """
    Check calculation of performance metrics function
    """
    X_train, y_train, val = train_dataset

    savepath = model_path
    if os.path.isfile(savepath):
        model = pickle.load(open(savepath, 'rb'))
        preds = inference(model, X_train)

        try:
            precision, recall, fbeta = compute_model_metrics(y_train, preds)
        except Exception as err:
            logging.error(
            "Performance metrics cannot be calculated on train data")
            raise err
    else:
        pass

def test_compute_confusion_matrix(train_dataset, model_path):
    """
    Check calculation of confusion matrix function
    """
    X_train, y_train, val = train_dataset

    savepath = model_path
    if os.path.isfile(savepath):
        model = pickle.load(open(savepath, 'rb'))
        preds = inference(model, X_train)

        try:
            cm = compute_confusion_matrix(y_train, preds)
        except Exception as err:
            logging.error(
            "Confusion matrix cannot be calculated on train data")
            raise err
    else:
        pass