"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020/11/25
- Brief: A class of Self-Organizing Map Network
"""

import numpy as np

class SOM():
    """ Self-Organizing Map Network """

    def __init__(self, input_dim, map_shape):
        """
        params:
        - input_dim: dimension of input
        - map_shape: shape after mapping
        """
        self.input_dim = input_dim
        self.map_shape = map_shape
        self.ws = np.random.randn(map_shape[0], map_shape[1], self.input_dim)

    def forward(self, xs):
        """
        params:
        - xs: input data, [batch size, number of attributes]
        """
        self.xs = xs

        # forward
        self.dists = np.array([np.sum((self.ws-x)**2, axis=-1) for x in self.xs])
        map_indices = np.argmin(self.dists.reshape(xs.shape[0], -1), axis=-1).reshape(-1, 1)
        self.map_coords = np.concatenate((map_indices//self.map_shape[1], map_indices%self.map_shape[1]), axis=-1)

        return self.map_coords

    def backward(self, update_radius):
        """ backward propagation """
        m, n = self.map_shape[0], self.map_shape[1]
        batch_size = self.xs.shape[0]
        batch_delta_ws = np.zeros(self.ws.shape)[np.newaxis, :].repeat(batch_size, axis=0)

        # calculate gradient
        lr = 1/(self.n_step+2)      # learning rate for nearest w
        for idx in range(batch_size):
            dists = np.array([[max(r-self.map_coords[idx][0], c-self.map_coords[idx][1]) for c in range(n)] for r in range(m)])
            dists = np.expand_dims(dists, axis=-1).repeat(self.input_dim, axis=-1)
            batch_delta_ws[idx][dists<=update_radius] = (np.exp(-dists)*lr*(self.ws-self.xs[idx]))[dists<=update_radius]
        self.delta_ws = np.sum(batch_delta_ws, axis=0)/batch_size

        # gradient descend
        self.ws -= self.delta_ws

    def learn(self, xs, steps, batch_size=16, update_radius=4):
        """ learn from xs """
        self.n_step = 0
        while self.n_step < steps:
            cnt = int(self.n_step/steps*100)
            print("\r|%s| "%("="*cnt+">"+"."*(99-cnt))+str(cnt+1)+"%", end="")
            batch_xs = xs[np.random.choice(xs.shape[0], batch_size)]
            _ = self.forward(batch_xs)
            self.backward(update_radius)
            self.n_step += 1
        print("\nEnd.\n")
