import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import *
from lightgbm import log_evaluation


class RemoveBadFeature(object):
    def __init__(self, model, X_train, y_train, X_valid, y_valid, 
                 deleted_features=[], top_n=10, verbose_period=500, n_repeats=5):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.model = model
        self.init_model = model
        self.deleted_features = deleted_features
        self.top_n = top_n
        self.n_repeats = n_repeats
        self.verbose_period = verbose_period

    def TrainModel(self, permutation=False):
        X_train_new = self.X_train.drop(self.deleted_features, axis=1)
        X_valid_new = self.X_valid.drop(self.deleted_features, axis=1)
        self.model.fit(X_train_new, self.y_train,
              eval_set=[(X_valid_new, self.y_valid)],
              eval_metric='rmse',
              callbacks=[log_evaluation(self.verbose_period)]
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

        #find bad features
        bad_feature_cnt = 0
        print(top_deleted)
        for name, mean in top_deleted:
            self.deleted_features.append(name)
            res, _ = self.TrainModel()
            self.deleted_features.pop()

            print('result: ', res, base_res)
            if res < base_res:
                self.deleted_features.append(name)
                base_res = res
                bad_feature_cnt += 1

        return bad_feature_cnt

    def Run(self):
        while True:
            bad_feature_cnt = self.RemoveOneFeature()
            if bad_feature_cnt == 0:
                break
        return

