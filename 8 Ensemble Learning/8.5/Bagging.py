"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2021.06.17
- Brief: A general Bagging class.
"""

import copy
import numpy as np

class Bagging(object):
    """ Bagging Algorithm """

    def __init__(self, base_learner, base_num):
        """ init base learners """
        self.base_num = base_num
        self.base_learners = [copy.deepcopy(base_learner) for i in range(self.base_num)]

    def fit(self, train_xs, train_ys):
        """ fit input training dataset """
        np.random.seed(0)
        for t in range(self.base_num):
            # create training dataset based on pdf of source dataset
            data_idx = np.random.choice(train_xs.shape[0], train_xs.shape[0], replace=True)
            selected_xs = train_xs[data_idx]
            selected_ys = train_ys[data_idx]
            # train one base learning
            self.base_learners[t].fit(selected_xs, selected_ys)

    def classify(self, x):
        """ classify x by voting """
        out = np.argmax(np.bincount([learner.classify(x) for learner in self.base_learners]))
        return out

    def test(self, test_xs, test_ys):
        """ test accuracy of input testing dataset """
        preds = [self.classify(test_x) for test_x in test_xs]
        isCorrect = preds == test_ys
        accuracy = sum(isCorrect)/len(isCorrect)
        return accuracy
