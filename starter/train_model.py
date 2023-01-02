"""
Author: Ali Kilinc
Date: 31.12.2022

This script is used to run whole training model process. It is feeded from side python files like model and data, to use functions created inside them. 
"""
# Script to train machine learning model.

from data import process_data
from model import train_model, compute_model_metrics, inference, compute_slices
from model import compute_confusion_matrix
from sklearn.model_selection import train_test_split
import pickle, os
import pandas as pd
import logging

def remove_if_exists(filename):
    """
    Delete a file if it exists in the file directory
    input:
        filename: str - path to the file to be removed
    output:
        None
    """
    if os.path.exists(filename):
        os.remove(filename)


# Initialize logging
logging.basicConfig(filename='train_model_logging.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Part that will fetch the data
csv_path = "/mnt/c/Users/tutkukilinc/nd0821-c3-starter-code/starter/data/census_modified.csv"
df = pd.read_csv(csv_path)

# Here I am splitting data into Train and Validation sets. Inside Train data
# I am going to use a GridSearchCV scheme
train, val = train_test_split(df, 
                                test_size=0.20, 
                                random_state=23, 
                                stratify=df['salary']
                                )

cat_features = ['workclass',
 'education',
 'marital-status',
 'occupation',
 'relationship',
 'race',
 'sex',
 'native-country']

X_train, y_train, cat_cols, num_cols, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Proces the test data with the process_data function.
# Set train flag = False - We use the encoding from the train set
# Setting like this, we are being sure that not leaking information into valid set
X_val, y_val, cat_cols_pass, num_cols_pass, lb_pass = process_data(
    val,
    categorical_features=cat_features,
    label="salary",
    training=False,
    lb=lb
)

# check if trained model already exists
model_savepth = "/mnt/c/Users/tutkukilinc/nd0821-c3-starter-code/starter/model"
filename = ['trained_model.pkl', 'labelizer.pkl']

# if saved model exits, load the model from disk
if os.path.isfile(os.path.join(model_savepth,filename[0])):
        model = pickle.load(open(os.path.join(model_savepth,filename[0]), 'rb'))
        lb = pickle.load(open(os.path.join(model_savepth,filename[1]), 'rb'))

# Else Train and save a model.
else:
    model = train_model(cat_cols, num_cols, X_train, y_train)
    # save model  to disk in ./model folder
    pickle.dump(model, open(os.path.join(model_savepth,filename[0]), 'wb'))
    pickle.dump(lb, open(os.path.join(model_savepth,filename[1]), 'wb'))
    logging.info(f"Model saved to disk: {model_savepth}")


# evaluate trained model on test set
preds = inference(model, X_val)
precision, recall, fbeta = compute_model_metrics(y_val, preds)

logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(
    f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")

cm = compute_confusion_matrix(y_val, preds, labels=list(lb.classes_))

logging.info(f"Confusion matrix:\n{cm}")

# Compute performance on slices for categorical features
# save results in a new txt file
slice_savepath = "/mnt/c/Users/tutkukilinc/nd0821-c3-starter-code/starter/model/slice_output.txt"
remove_if_exists(slice_savepath)

# iterate through the categorical features and save results to log and txt file
for feature in cat_features:
    performance_df = compute_slices(val, feature, y_val, preds)
    performance_df.to_csv(slice_savepath,  mode='a', index=False)
    logging.info(f"Performance on slice {feature}")
    logging.info(performance_df)