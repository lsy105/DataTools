from lightgbm import *



class RegModel(object):
    def __init__(self, cv):
        self.cv = cv
        self.model = None

    def Train(self, X_train, y_train, X_test=None, y_test=None):
        model = LGBMRegressor(
                  objective="regression",
                  metric="rmse",
                  boosting_type="gbdt",
                  n_estimators=1,
                  num_leaves=20,
                  max_depth=5,
                  learning_rate=0.05,
                  subsample=0.8,
                )

        eval_data = None
        if X_test is not None and y_test is not None:
            eval_data = [(X_test, y_test)]

        model.fit(X_train, y_train,
                  eval_set=eval_data,
                  eval_metric='rmse',
                 )
        return model

    def fit(self, X, y):
        res = 0.0
        cnt = 0
        from collections import defaultdict
        mp = defaultdict(int)
        for train_index, test_index in self.cv.split(X):
            self.model = self.Train(X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index])
            res += self.model.best_score_['valid_0']['rmse']
            for i in range(len(self.model.feature_importances_)):
                mp[self.model.feature_name_[i]] += self.model.feature_importances_[i]
            cnt += 1
        for key in mp:
            print(key, mp[key])
        return res / cnt
