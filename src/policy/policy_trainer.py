from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from lightgbm import *


class PolicyTrainer(object):
    def __init__(self, optimizer, policy_model, model, X, y, dataset_builder, feature_list, entropy_parameter):
        self.optimizer = optimizer
        self.policy_model = policy_model
        self.model = model
        self.entropy_parameter = entropy_parameter
        self.X = X
        self.y = y
        self.feature_list = feature_list
        self.dataset_builder = dataset_builder
        self.baseline = None

    def GetLoss(self):
        log_prob, entropy, feature_idxes = self.policy_model()

        X_new = self.dataset_builder.Build(self.X, self.feature_list, feature_idxes)
        import re
        X_new = X_new.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        print(feature_idxes)
        env_reward = 0.0 - self.model.fit(X_new, self.y)
        del X_new

        reward = env_reward #+ self.entropy_parameter * entropy

        #baseline (neg_reward - baseline)
        if self.baseline is None:
            self.baseline = reward
        else:
            x = 0.99
            self.baseline = x * self.baseline + (1.0 - x) * reward
        #self.baseline = self.baseline.detach()

        print(log_prob, reward, self.baseline)
        loss = 100.0 * log_prob * (reward - self.baseline)
        return loss

    def Train(self, num_epochs):
        update_w = 1
        for iter_id in range(num_epochs):
            loss = self.GetLoss()
            loss /= update_w
            loss.backward(retain_graph=True)
            print(loss)
            if iter_id % update_w == 0 and iter_id > 0:
                grad_norm_normal = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 0.25)
                self.optimizer.step()
                self.optimizer.zero_grad()


