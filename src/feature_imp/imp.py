import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import *


class RemoveBadFeature(object):
    def __init__(self, model, X_train, y_train, X_valid, y_valid, deleted_features=[], top_n=10, n_repeats=5):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.model = model
        self.init_model = model
        self.deleted_features = deleted_features
        self.top_n = top_n
        self.n_repeats = n_repeats

    def TrainModel(self, permutation=False):
        X_train_new = self.X_train.drop(self.deleted_features, axis=1)
        X_valid_new = self.X_valid.drop(self.deleted_features, axis=1)
        self.model.fit(X_train_new, self.y_train,
              eval_set=[(X_valid_new, self.y_valid)],
              eval_metric='rmse',
              #callbacks=[early_stopping(10, verbose=False), log_evaluation(10)]
             )

        top_deleted = []
        if permutation:
            r = permutation_importance(self.model, X_valid_new, self.y_valid,
                                       n_repeats=self.n_repeats,
                                       random_state=0)
            for i in range(len(r.importances_mean)):
                if r.importances_mean[i] < 0.0:
                    top_deleted.append((X_valid_new.columns[i], r.importances_mean[i]))
            top_deleted = sorted(top_deleted, key=lambda x:x[1])[:self.top_n]

        res = self.model.best_score_['valid_0']['rmse']
        self.model = self.init_model

        return res, top_deleted

    def RemoveOneFeature(self):
        base_res, top_deleted = self.TrainModel(True)

        #find the worst feature
        worst_feature = []
        print(top_deleted)
        for name, mean in top_deleted:
            self.deleted_features.append(name)
            res, _ = self.TrainModel()
            self.deleted_features.pop()

            if res < base_res:
                if len(worst_feature) == 0 or res < worst_feature[0][1]:
                    worst_feature = [(name, res)]
        return worst_feature 

    def Run(self):
        while True:
            worst_feature = self.RemoveOneFeature()
            if len(worst_feature) == 0:
                break
            self.deleted_features.append(worst_feature[0][0])
        return

