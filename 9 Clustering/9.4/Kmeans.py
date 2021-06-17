"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020/11/23
- Brief: K-means Algorithm
"""

import numpy as np

class Kmeans():
    """ Cluster Algorithm -- K-means """

    def __init__(self, k):
        self.k = k                  # parameter of K-means

    def cluster(self, xs):
        """ k-means algorithm """
        xs = xs.copy()
        np.random.shuffle(xs)       # randomly
        self.means = xs[:self.k]    # init k mean vectors

        np.random.shuffle(xs)
        stop = False
        while not stop:
            # init clusters
            clusters = [np.empty(shape=[0, xs.shape[1]]) for i in range(self.k)]
            # put each x into nearest cluster
            for j in range(xs.shape[0]):
                nearest_idx = np.argmin(np.sum((self.means-xs[j])**2, axis=1))
                clusters[nearest_idx] = np.append(clusters[nearest_idx], [xs[j]], axis=0)
            # calculate new means
            new_means = np.array([np.mean(clusters[i], axis=0) for i in range(self.k)])
            stop = (new_means == self.means).all()
            self.means = new_means

        return np.array(clusters), self.means