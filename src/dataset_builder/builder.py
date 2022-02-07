import pandas as pd
import numpy as np

class DatasetBuilder(object):
    def GenFeature(self, name, X, X_new):
        A, O, B = name.split(',')
        print(A, O, B)
        if A == B == 'zero':
            return

        if A == 'zero':
            X_new[B] = X[B]
            return
        if B == 'zero':
            X_new[A] = X[A]
            return

        if O == '/':
            O = 'D'
            X_new[name + O] = X[A] / (X[B] + 1e-6)
        elif O == '+':
            O = 'P'
            X_new[name + O] = X[A] + X[B]
        elif O == '-':
            O = 'M'
            X_new[name + O] = X[A] - X[B]
        elif O == '*':
            O = 'T'
            X_new[name + O] = X[A] * X[B]
        else:
            return

    def Build(self, X, feature_list, op_list, feature_idxes):
        X_new = pd.DataFrame()
        for A_idx, op_idx, B_idx in feature_idxes:
            A_name = feature_list[A_idx]
            op_name = op_list[op_idx]
            B_name = feature_list[B_idx]
            name = A_name + ',' + op_name + ',' + B_name 
            self.GenFeature(name, X, X_new)
        return X_new

            
        
         

    

