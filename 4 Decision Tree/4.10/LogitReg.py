"""
- Author: Haoxin Lin
- Email: linhx36@outlook.com
- Date: 2020.09.20
- Brief: Logit Regression
"""

import numpy as np
import matplotlib.pyplot as plt

class LogitReg:
    """ Logistic Regression """

    def __init__(self, xsize):
        """ Init parameters of the learner """
        # xsize: size of input
        self.xsize = xsize
        self.w = np.random.randn(self.xsize)
        self.b = np.random.randn()
        self.lr = 0.1      # learning rate

    def reset(self):
        """ Reset parameters """
        self.w = np.random.randn(self.xsize)
        self.b = np.random.randn()
        self.lr = 0.1
        del self.xs, self.ys

    def load(self, xs, ys):
        """ load dataset """
        # xs: x data
        # ys: y data
        self.xs = xs
        self.ys = ys

    def loss(self):
        """ Calculate Loss """
        ls = 0
        for i in range(self.xs.shape[0]):
            y = self.w.dot(self.xs[i]) + self.b    # output
            ls += np.log(1+np.exp(y)) - self.ys[i]*y
        ls /= self.xs.shape[0]
        return ls

    def learn(self, steps, visible=True):
        """ Learn nearly optimal parameters from data """
        if visible:
            print("\n### Start learning ###")
            print("Initial loss: %.4f"%self.loss())

        for step in range(steps):
            # 1st, 2st gradient of loss
            onegrad2w, twograd2w = np.zeros(self.w.shape), 0
            onegrad2b, twograd2b = 0, 0

            # gradient descent
            for i in range(self.xs.shape[0]):
                y = self.w.dot(self.xs[i]) + self.b    # output
                if y < 0:
                    p1 = np.exp(y)/(1+np.exp(y))      # likelihood func of y = 1
                else:
                    p1 = 1/(1+np.exp(-y))
                onegrad2w += self.xs[i]*(p1-self.ys[i])
                onegrad2b += p1 - self.ys[i]
                twograd2w += self.xs[i].dot(self.xs[i])*p1*(1-p1)
                twograd2b += p1*(1-p1)
            if twograd2w != 0:
                self.w -= self.lr*onegrad2w/twograd2w
            else:
                self.w = np.random.randn(self.xsize)
            if twograd2b != 0:
                self.b -= self.lr*onegrad2b/twograd2b
            else:
                self.b = np.random.randn()

            # new loss
            if step % 1000 == 0 and visible:
                print("Step %d: loss=%.4f"%(step+1, self.loss()))

    def predict(self, input):
        """ Predict new input """
        y = self.w.dot(input) + self.b
        return 1 if y > 0 else 0

    def test(self, xs, ys):
        """ Test on test dataset """
        prediction = np.array([self.predict(x) for x in xs])
        acc = np.sum(prediction==ys)/len(ys)
        return acc