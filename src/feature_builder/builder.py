import numpy as np
import pandas as pd

class FeatureBuilder(object):
    def __init__(self, feature_candidates, feature_ops):
        self.feature_candidates = feature_candidates
        self.feature_ops = feature_ops
        self.feature_list = []

    def Create(self):
        N = len(self.feature_candidates)
        M = len(self.feature_ops) 
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(M):
                    temp = self.feature_candidates[i] + ',' + self.feature_ops[k] + ',' + self.feature_candidates[j]
                    self.feature_list.append(temp)






