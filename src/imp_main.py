import numpy as np
import pandas as pd
from feature_imp.imp import *

from lightgbm import *
from sklearn.model_selection import KFold


if __name__ == '__main__':
    df = pd.read_parquet('train_low_mem.parquet').sample(frac=0.1)
    X = df.drop(['target', 'row_id', 'time_id'], axis=1)
    y = df['target']
    del df

    model = LGBMRegressor(
               objective="regression",
               metric="rmse",
               boosting_type="gbdt",
               n_estimators=200,
               num_leaves=100,
               max_depth=10,
               learning_rate=0.05,
               subsample=0.8,
            ) 
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    remove_obj = RemoveBadFeature(model, X_train, y_train, X_valid, y_valid, deleted_features=[], top_n=5, n_repeats=5)
    remove_obj.Run()

