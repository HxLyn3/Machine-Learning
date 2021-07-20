"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2021/01/23
- Brief: A class of naive Bayes classifier
"""

import math
import numpy as np
from tqdm import tqdm

class NaiveBayesClassifier():
    """ naive Bayes classifier (with Laplacian correction) """

    def __init__(self, xs, isdiscs, ys):
        self.xs = xs                                # x data
        self.isdiscs = isdiscs                      # a vector, 1 means the corresponding attribute is discrete
        self.ys = ys                                # y data

        self.n_attributes = xs.shape[1]             # number of attributes
        self.n_classes = len(set(self.ys))          # number of classes
        self.classes = list(set(self.ys))           # class enumerate

        # calculate class prior probability (with Laplacian correction)
        self.prob_c = [(len(self.ys[self.ys==clss])+1)/(self.ys.shape[0]+self.n_classes) for clss in self.classes]

        # divide xs by class
        self.xs_by_class = [self.xs[self.ys==clss] for clss in self.classes]

        # calculate conditional probability for each attribute
        self.cond_prob_table = []
        for i in range(self.n_attributes):
            self.cond_prob_table.append({})         # init conditional probability of this attribute
            if self.isdiscs[i]:                     # if the attribute is discrete
                attri_values = list(set(self.xs[:, i]))

            for cls_idx in range(self.n_classes):
                sub_xs = self.xs_by_class[cls_idx]  # xs in this class
                self.cond_prob_table[i][self.classes[cls_idx]] = {}
                if self.isdiscs[i]:
                    for val in attri_values:
                        count = len(sub_xs[sub_xs[:, i]==val])
                        # conditional probability with Laplacian correction
                        self.cond_prob_table[i][self.classes[cls_idx]][val] = (count+1)/(len(sub_xs)+len(attri_values))
                else:
                    self.cond_prob_table[i][self.classes[cls_idx]]['mu'] = np.mean(sub_xs[:, i].astype(np.float64))
                    self.cond_prob_table[i][self.classes[cls_idx]]['sigma'] = np.std(sub_xs[:, i].astype(np.float64))

    def classify(self, x):
        """ classify x """
        post_probs_c = [1]*self.n_classes
        for cls_idx in range(self.n_classes):
            post_probs_c[cls_idx] *= self.prob_c[cls_idx]
            for i in range(self.n_attributes):
                if self.isdiscs[i]:                 # if the attribute is discrete
                    post_probs_c[cls_idx] *= self.cond_prob_table[i][cls_idx][x[i]]
                else:
                    mu = self.cond_prob_table[i][cls_idx]['mu']
                    sigma = self.cond_prob_table[i][cls_idx]['sigma']
                    post_probs_c[cls_idx] *= (1/(math.sqrt(2*math.pi)*sigma))*math.exp(-(x[i].astype(np.float64)-mu)**2/(2*sigma**2))
        return self.classes[np.argmax(np.array(post_probs_c))]

    def test(self, xs):
        ys = np.array([self.classify(xs[i]) for i in tqdm(range(xs.shape[0]))])
        return ys