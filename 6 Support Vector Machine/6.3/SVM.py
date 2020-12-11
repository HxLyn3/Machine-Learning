"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020/12/06
- Brief: A SVM class.
"""

from tqdm import tqdm
import numpy as np

class SVM():
    """ Support Vector Machine """

    def __init__(self, input_dim, **kernel):
        self.input_dim = input_dim
        self.kernel = kernel

    def Kernel_func(self, xi, xj):
        """ kernel function """
        out = None
        if self.kernel['func'] == 'Linear':
            out = xi.dot(xj)
        elif self.kernel['func'] == 'Polynomial':
            out = (xi.dot(xj))**self.kernel['d']
        elif self.kernel['func'] == 'Gaussian':
            out = np.exp(-np.sum((xi-xj)**2)/(2*self.kernel['sigma']**2))
        elif self.kernel['func'] == 'Lapras':
            out = np.exp(-np.sum(np.abs(xi-xj))/self.kernel['sigma'])
        elif self.kernel['func'] == 'Sigmoid':
            out = np.tanh(self.kernel['beta']*xi.dot(xj)+self.kernel['theta'])
        return out

    def fit(self, xs, ys, C=0.6, epsilon=0.001, iters=100):
        """ train SVM with dataset (xs, ys) with SMO algorithm """
        num = xs.shape[0]
        self.dataMatrix = xs.copy()
        self.labelVec = ys.copy()
        # kernel matrix, K_matrix[i][j] = K(x[i], x[j])
        K_matrix = np.array([[self.Kernel_func(xs[i], xs[j]) for j in range(num)] for i in range(num)])

        # init parameters
        self.alphas = np.zeros(num)
        self.b = 0

        for it in tqdm(range(iters)):
            for i in range(num):
                # calculate f(xi) and Ei
                fx_i = np.sum(self.alphas*ys*K_matrix[i]) + self.b
                E_i = fx_i - ys[i]

                # if alpha_i don't satisfy KKT condition (with a little error-tolerant rate)
                if (ys[i]*E_i < -epsilon and self.alphas[i] < C) or (ys[i]*E_i > epsilon and self.alphas[i] > 0):
                    # select j randomly
                    probs = np.ones(num)/(num-1)
                    probs[i] = 0
                    j = np.random.choice(range(num), p=probs)
                    # calculate f(xj) and Ej
                    fx_j = np.sum(self.alphas*ys*K_matrix[j]) + self.b
                    E_j = fx_j - ys[j]

                    # old alpha i and alpha j
                    alpha_i_old, alpha_j_old = self.alphas[i].copy(), self.alphas[j].copy()

                    # calculate lower bound and upper bound of alpha j
                    if ys[i] != ys[j]:
                        zeta = alpha_i_old - alpha_j_old
                        L, H = max(0, -zeta), min(C, C-zeta)
                    else:
                        zeta = alpha_i_old + alpha_j_old
                        L, H = max(0, zeta-C), min(C, zeta)
                    if L == H: continue

                    # calculate eta
                    eta = K_matrix[i][i] + K_matrix[j][j] - 2*K_matrix[i][j]
                    if eta <= 0: continue

                    # update and clip alpha j
                    self.alphas[j] += ys[j]*(E_i-E_j)/eta
                    self.alphas[j] = min(H, max(L, self.alphas[j]))
                    # update alpha i
                    self.alphas[i] += ys[i]*ys[j]*(alpha_j_old-self.alphas[j])

                    # update b
                    b1 = self.b - E_i - ys[i]*(self.alphas[i]-alpha_i_old)*K_matrix[i][i] \
                        - ys[j]*(self.alphas[j]-alpha_j_old)*K_matrix[j][i]
                    b2 = self.b - E_j - ys[j]*(self.alphas[i]-alpha_i_old)*K_matrix[i][j] \
                        - ys[j]*(self.alphas[j]-alpha_j_old)*K_matrix[j][j]
                    if self.alphas[i] > 0 and self.alphas[i] < C: self.b = b1
                    elif self.alphas[j] > 0 and self.alphas[j] < C: self.b = b2
                    else: b = (b1+b2)/2

        self.support_vectors = []
        for i in range(num): 
            if abs(abs(self._f(xs[i]))-1) <= 0.01: self.support_vectors.append(xs[i])
        self.support_vectors = np.array(self.support_vectors)

    def _f(self, input_x):
        """ f(x) """
        Kernel_outs = np.array([self.Kernel_func(self.dataMatrix[i], input_x) for i in range(self.dataMatrix.shape[0])])
        out = np.sum(self.alphas*self.labelVec*Kernel_outs) + self.b
        return out

    def predict(self, input_xs):
        """ predict class of input_xs """
        clss = np.array([1 if self._f(input_xs[i]) > 0 else -1 for i in tqdm(range(input_xs.shape[0]))])
        return clss