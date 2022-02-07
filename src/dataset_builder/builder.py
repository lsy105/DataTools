import pandas as pd
import numpy as np

class DatasetBuilder(object):
    def GenFeature(self, name, X):
        print(name)
        A, O, B = name.split(',')
        if O == '/':
            O = 'D'
            X[name + O] = X[A] / (X[B] + 1e-6)
        elif O == '+':
            O = 'P'
            X[name + O] = X[A] + X[B]
        elif O == '-':
            O = 'M'
            X[name + O] = X[A] - X[B]
        elif O == '*':
            O = 'T'
            X[name + O] = X[A] * X[B]
        else:
            return

    def Build(self, X, feature_list, feature_idxes):
        X_new = X.copy() 
        for idx in feature_idxes:
            name = feature_list[idx] 
            self.GenFeature(name, X_new)
        return X_new

            
        
         

    

