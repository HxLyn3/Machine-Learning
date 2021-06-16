"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2021.06.16
- Brief: A general AdaBoost class.
"""

import copy
import numpy as np

class AdaBoost(object):
    """ AdaBoost Algorithm """

    def __init__(self, base_learner, base_num):
        """ init base learners """
        self.base_num = base_num
        self.base_learners = [copy.deepcopy(base_learner) for i in range(self.base_num)]
        self.weights = [0]*self.base_num

    def fit(self, train_xs, train_ys):
        """ fit input training dataset """

        data_num = train_xs.shape[0]                        # number of samples in training dataset
        data_pdf = np.full(data_num, 1/data_num)            # probability distribution func of training samples

        for t in range(self.base_num):
            # create adjusted training dataset (1000 samples) based on pdf of source dataset
            data_idx = np.random.choice(data_num, 1000, replace=True, p=data_pdf)
            adjusted_xs = train_xs[data_idx]
            adjusted_ys = train_ys[data_idx]

            # train one base learning, and calculate its corresponding weight
            self.base_learners[t].fit(adjusted_xs, adjusted_ys)
            accuracy = self.base_learners[t].test(adjusted_xs, adjusted_ys)
            self.weights[t] = 0.5*np.log(accuracy/(1-accuracy))

            # adjust distribution of dataset
            for idx in range(data_num):
                data_pdf[idx] *= np.exp(-self.weights[t]*int(self.base_learners[t].classify(train_xs[idx]) == train_ys[idx]))
            data_pdf /= np.sum(data_pdf)

        # normalize weights
        self.weights /= np.sum(self.weights)

    def classify(self, x):
        """ classify x with weighted base learners """
        out = np.sum(np.array(self.weights)*np.array([learner.classify(x) for learner in self.base_learners]))
        return round(out)

    def test(self, test_xs, test_ys):
        """ test accuracy of input testing dataset """
        preds = [self.classify(test_x) for test_x in test_xs]
        isCorrect = preds == test_ys
        accuracy = sum(isCorrect)/len(isCorrect)
        return accuracy


