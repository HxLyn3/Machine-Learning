"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020/11/23
- Brief: A class of monolayer RBF(Radial Basis Function) Network
"""

import numpy as np
from Kmeans import Kmeans

def softmax(inputs):
    """ activation function: softmax """
    exp_outs = np.exp(inputs)
    exp_sums = np.sum(exp_outs, axis=1)
    return exp_outs/(np.expand_dims(exp_sums, axis=1).repeat(exp_outs.shape[1], axis=1))

class MonoRBF():
    """ Monolayer RBF """

    def __init__(self, hidden_size, n_classes):
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.ws = np.random.randn(self.hidden_size, self.n_classes)/np.sqrt(self.hidden_size)
        self.betas = np.random.randn(self.hidden_size)

    def load_training(self, xs, ys):
        self.train_xs = xs
        self.train_ys = ys

        # cluster, then generate centers
        _, self.cs = Kmeans(self.hidden_size).cluster(self.train_xs)

    def forward(self, xs):
        """ phi(x) = Sigma(w[i]*exp(-β[i]*|x-c[i]|²)) """
        self.ls = np.array([np.sum((x-self.cs)**2, axis=1) for x in xs])
        self.hs = self.ls*self.betas
        self.hs_act = np.exp(-self.hs)
        self.ys = np.matmul(self.hs_act, self.ws)
        self.preds = softmax(self.ys)
        return self.preds

    def _loss(self, preds, ys):
        """ calculate loss """
        cross_entropy = -np.mean(np.log(preds[range(preds.shape[0]), ys]))
        return cross_entropy

    def backward(self, ys):
        """ Backward Propogation """
        # batch size
        bs = self.preds.shape[0]

        # loss's gradient to output of RBF Network
        self.delta_ys_softmax = np.zeros(self.preds.shape)
        self.delta_ys_softmax[range(bs), ys] = -1/(bs*self.preds[range(bs), ys])

        # gradient of softmax function
        actfunc_grads = [None for i in range(bs)]
        for idx in range(bs):
            actfunc_grads[idx] = -np.matmul(self.preds[idx].reshape(-1, 1), self.preds[idx].reshape(1, -1))
            actfunc_grads[idx] += np.diag(self.preds[idx])
        actfunc_grads = np.array(actfunc_grads)

        self.delta_ys = np.einsum('ijk, ikl -> ijl', self.delta_ys_softmax.reshape(bs, 1, -1), actfunc_grads).reshape(bs, -1)

        # loss's gradient
        self.delta_ws = np.matmul(self.hs_act.T, self.delta_ys)             # to ws of RBF Network
        self.delta_hs_act = np.matmul(self.delta_ys, self.ws.T)             # to hs after activation
        self.delta_hs = self.delta_hs_act*(-self.hs_act)                    # to hs of RBF Network
        self.delta_betas = np.sum(self.delta_hs*self.ls, axis=0)            # to betas of RBF Network

    def learn(self, lr, epochs):
        """ train RBF Network """
        den = 100
        for epoch in range(epochs):
            num = int(epoch/epochs*den)
            preds = self.forward(self.train_xs)
            loss = self._loss(preds, self.train_ys)
            print("\r|%s| loss=%.4f"%("="*num+">"+"."*(den-num-1), loss), end='', flush=True)
            self.backward(self.train_ys)

            # update
            self.ws -= lr*self.delta_ws
            self.betas -= lr*self.delta_betas
        print("\nEnd.\n")