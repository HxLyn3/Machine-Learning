import numpy as np

class kNN(object):
    """ an implementation of k-Nearest Neighbor """

    def __init__(self, k):
        self.k = k

    def fit(self, train_xs, train_ys):
        """ just load dataset """
        self.xs = train_xs
        self.ys = train_ys

    def classify(self, x):
        """ predict x """
        dist = np.sum(np.abs(self.xs-x), axis=1)
        sort_idx = np.argsort(dist)
        k_labels = self.ys[sort_idx][:self.k]
        return np.argmax(np.bincount(k_labels))

    def test(self, test_xs, test_ys):
        """ test accuracy of input testing dataset """
        preds = [self.classify(test_x) for test_x in test_xs]
        isCorrect = preds == test_ys
        accuracy = sum(isCorrect)/len(isCorrect)
        return accuracy