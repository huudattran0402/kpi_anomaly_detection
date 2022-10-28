import numpy as np
from CellPAD.feature import FeatureTools
from CellPAD.preprocessor import Preprocessor 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score


def build_model(timestamps, series, to_remove_trend):

    if not to_remove_trend:
        processed_series = np.array(series)
    else:
        preprocessor = Preprocessor()
        trend_remove_method="past_mean"
        period_len = 168
        processed_series = preprocessor.remove_trend(series, period_len, method=trend_remove_method)

    feature_types=["Indexical", "Numerical"]
    feature_time_grain=["Weekly"]
    feature_operations=["Wma", "Ewma", "Mean", "Median"]
    featureTools = FeatureTools()
    feature_list = featureTools.set_feature_names(feature_types, feature_time_grain, feature_operations)

    train_test_features = featureTools.compute_feature_matrix(
                                                                timestamps = timestamps,
                                                                series = processed_series,
                                                                labels=[False] * len(timestamps),
                                                                ts_period_len = 168,
                                                                feature_list = feature_list,
                                                                start_pos=0,
                                                                end_pos = len(series)
                                                            )
    X_train, X_test, y_train, y_test = train_test_split(train_test_features[168:], processed_series[168:])
    tree = DecisionTreeRegressor().fit(X_train, y_train)
    metrics = {
        "train_score":evaluate_regression_model(tree, X_train, y_train),
        "test_score":evaluate_regression_model(tree, X_test, y_test)
    }
    return tree, metrics,train_test_features
    
def evaluate_regression_model(model,x,y):
    '''
        - R Square is a good measure to determine how well the model fits the dependent variables. 
            However, it does not take into consideration of overfitting problem.
        - Mean Square Error gives you an absolute number on how much your predicted results deviate from the actual number.
        - Compare to MSE or RMSE, MAE is a more direct representation of sum of error terms. 
            MSE gives larger penalization to big prediction error by square it while MAE treats all errors the same.
    '''
    y_predicted = model.predict(x)
    dict = {
        "r2_score": r2_score(y, y_predicted),
        "mean_squared_error": mean_squared_error(y, y_predicted),
        "Mean_absolute_error": mae(y, y_predicted)
    }
    return dict
