import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=[], label=None, training=True,lb=None
):
    """ Process the data used in the machine learning pipeline.

    This process data section is used only to labelize Y column. Other preprocessing steps are applied into GridSearchCV part inside model_training function.

    Also cat columns and num columns are defined here. It can be set in many different ways, but for this simple scenario i am not digging deeper.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    cat_cols: List
        List of categorical columns inside data
    num_cols: List
        List of numerical columns inside data
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    cat_cols = categorical_features
    num_cols = [col for col in X.columns if col not in cat_cols]
    
    if training is True:
        lb = LabelBinarizer()
        y = lb.fit_transform(y.values).ravel()
    else:
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    return X, y, cat_cols, num_cols, lb