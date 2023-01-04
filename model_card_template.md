# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is using Census data to predict whether a person is earning >50K or <=50K, simply turning the earnings prediction into a classification problem. 

We used a LightGBMClassifier model to train the model. Also a GridSearchCV is used to find best hyperparameter set for this model. 

These are the used hyperparameter set for GridSearch section: 
	parameters = {
            'est__max_depth':[5, 9, 18],
            'est__learning_rate': [0.01, 0.1],
            'est__n_estimators': [10, 50, 100],
            'est__num_leaves': np.linspace(11,51,3,endpoint = True, dtype = int)
            }
			
Dataset is split into 2 parts initially, Training data and Validation data. Validation data is used as a hold-out set. After training finding the best model, we calculated modelling metrics
with this hold out set again, on this best model. 

Training data is ingested into GridSearch CV, a grid search object with CV. For this process, also a Pipeline object is used. Inside pipeline object,
MinMax Scaler is used for numerical values and OneHotEncoder for categorical values. Also, the target value was labelized before this pipeline. 

Using estimator and preprocessor objects in a Pipeline object together helps us to make transformations on train data only. So that test data for K-Folds will not get affected
from train data, which can lead to data leakage. 

Model and Labelizer objects were saved as pickle files, and all the model results were saved as a log file. 

## Intended Use

This model has intention of predicting the salary bracket of a person, by ingesting the census data as input features. 

## Training Data

The dataset used for this model is taken from The Census Income Dataset was obtained from the UCI Machine Learning Repository. It has 15 columns, 'salary' column as 
target feature together with 8 categorical and 6 numerical input features. 

Census Data: (https://archive.ics.uci.edu/ml/datasets/census+income)

This dataset has some problems, which we detected prior to developing a model, through census_eda.ipynb file. We made some transformations on dataset with this EDA file and 
saved the modified version of dataset as census_modeified.csv

At first, we used a train-test split of 80-20; also using a stratify on target variable. 

## Evaluation Data

For the 20% of data set aside as evaluation data, we used the best model object to make predictions, and also the label encoder object to make same transformations to 
target variable. Best model object, also makes the transformation on this dataset, applying the transformation rules generated on training data. 

## Metrics

_Please include the metrics used and your model's performance on those metrics._

Below results are from K-Fold CV for best model with best hyperparameters

Best_Hyperparameters: {'est__learning_rate': 0.1, 'est__max_depth': 18, 'est__n_estimators': 100, 'est__num_leaves': 31}
Average Train/Test Accuracy for K-Folds: train_accuracy = 0.893734647452599, test_accuracy = 0.8735793615730045
Average Train/Test ROC for K-Folds: 0.9521700959316386, test_roc = 0.9278138053302992
Average Train/Test F1 for K-Folds: train_f1 = 0.761629113861547, test_f1 = 0.7149982963872898
Average Train/Test Precision for K-Folds: train_precision = 0.8282170550980537, test_precision = 0.7821099424224556
Average Train/Test Recall for K-Folds: train_recall = 0.7049656105523601, test_recall = 0.6585342203752772

Below metrics are from Validation set hold-out at the beginning of process.

Precision on validation set:0.781
Recall on validation set:0.651
Fbeta on validation set:0.710

Confusion matrix on validation set:
[[4659  286]
 [ 547 1021]]

## Ethical Considerations

When we look at prediction performances on slices, they seem like balanced. So, we can say that predictive power of model is not discriminative on different population groups. 

We should consider these metrics also, in order to prevent any bias towards a population group. 

## Caveats and Recommendations

A better census dataset can be used to develop a similar model.

Also, technically, different algorithms than LightGBM can be used to make a classification model. 