import numpy as np
import pandas as pd

from feature_builder.builder import FeatureBuilder
from dataset_builder.builder import DatasetBuilder
from policy.policy_model import *  
from policy.policy_trainer import *  
from policy.policy_trainer import *  
from model.model import *
from lightgbm import *
from sklearn.model_selection import KFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


deleted_features = ['f_22', 'f_78', 'f_25', 'f_174', 'f_258']

#X_new = db.Builder(df, fb.feature_list, [8, 20])


if __name__ == '__main__':
    kf = KFold(n_splits=2)
    df = pd.read_parquet('train_low_mem.parquet').sample(frac=0.1)
    X = df.drop(['target', 'row_id', 'time_id'] + deleted_features, axis=1)
    y = df['target']
    del df

    model = RegModel(cv=kf)
    #loss = model.fit(X, y)
    #print('baseline loss:', loss)
    #imp_features = list(zip(model.model.feature_importances_, X.columns))
    #imp_features.sort(reverse=True)
    #print(imp_features[:10])
    op_list = ['+', '-', '/', '*']
    feature_list = X.columns.tolist() + ['zero']

    db = DatasetBuilder()
    pm = PolicyModel(num_layer=10, num_ops=len(op_list), num_features=len(feature_list), lstm_size=128, 
                    lstm_num_layers=2, tanh_constant=2.5, temperature=5)

    optimizer = torch.optim.Adam(pm.parameters(), lr=0.001)
    PT = PolicyTrainer(optimizer, pm, model, X, y, db, feature_list, op_list, 0.00001)
    PT.Train(20000)

