"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.11.08
- Brief: A Neural Network Class
"""

import numpy as np
from graphviz import Digraph

def relu(inputs):
    """ activation function: relu """
    return np.where(inputs < 0, 0, inputs)

def sigmoid(inputs):
    """ activation function: sigmoid """
    return np.where(inputs < 0, np.exp(inputs)/(1+np.exp(inputs)), 1/(1+np.exp(-inputs)))

def tanh(inputs):
    """ activation function: tanh """
    return (np.exp(inputs)-np.exp(-inputs))/(np.exp(inputs)+np.exp(-inputs))

def softmax(inputs):
    """ activation function: softmax """
    exp_outs = np.exp(inputs)
    exp_sums = np.sum(exp_outs, axis=1)
    return exp_outs/(np.expand_dims(exp_sums, axis=1).repeat(exp_outs.shape[1], axis=1))

class NN():
    """ Neural Network """

    def __init__(self, ndims, activation_functions, lr_init=0.0001, lr_decay=1, lr_min=0.0001, \
        regularization="L2", regularization_lambda=0.01, task_type="Classification"):
        """ Init Neural Network """
        self.ndims = ndims                  # [input_dim, hidden1_dim, hidden2_dim, ..., output_dim]
        self.nlayers = len(self.ndims) - 1
        # weights
        self.Ws = [np.random.randn(self.ndims[i], self.ndims[i+1])/np.sqrt(self.ndims[i]/2) for i in range(self.nlayers)]
        # biases
        self.bs = [np.zeros((1, self.ndims[i+1])) for i in range(self.nlayers)]

        # activation functions for each layers
        self.activation_functions = activation_functions

        # learning rate
        self.lr_init = lr_init
        self.lr = self.lr_init
        self.lr_decay = lr_decay
        self.lr_min = lr_min

        # regularization
        self.regularization = regularization
        self.lamb = regularization_lambda

        self.task_type = task_type

    def reset(self):
        # weights
        self.Ws = [np.random.randn(self.ndims[i], self.ndims[i+1]) for i in range(self.nlayers)]
        # biases
        self.bs = [np.random.randn(1, self.ndims[i+1]) for i in range(self.nlayers)]

        # learning rate
        self.lr = self.lr_init

    def forward(self, xs):
        """ forward propagation """
        self.layer_outs = []

        for layer_idx in range(self.nlayers):
            this_layer_out = {}

            # sent datas to neurons
            hs = np.matmul(xs, self.Ws[layer_idx]) + self.bs[layer_idx]
            this_layer_out['BeforeActivation'] = hs
            # activation
            activation_func = self.activation_functions[layer_idx]
            ys = eval(activation_func)(hs) if activation_func else hs
            this_layer_out['AfterActivation'] = ys

            self.layer_outs.append(this_layer_out)
            # output of this layer is the input of next layer
            xs = ys

        return self.layer_outs[-1]['AfterActivation']

    def _loss(self, preds, ys, loss_type="CrossEntropy"):
        """ calculate loss """
        bs = preds.shape[0]
        loss = None
        # use cross-entropy in classification tasks
        if loss_type == "CrossEntropy":
            cross_entropy = -np.mean(np.log(preds[range(preds.shape[0]), ys]))
            loss = cross_entropy
        # use mean square error in regression tasks
        elif loss_type == "MSE":
            mse = np.mean((preds-ys.reshape(-1, 1))**2)
            loss = mse

        # regularization
        if self.regularization == "L1":
            loss += self.lamb*sum([np.sum(np.abs(W)) for W in self.Ws])/bs
        elif self.regularization == "L2":
            loss += self.lamb*sum([np.sum(np.square(W)) for W in self.Ws])/bs

        return loss

    def backward(self):
        """ backward propogation """

        # batch size
        bs = self.preds.shape[0]

        # loss's gradient to output of neural network
        if self.loss_type == "CrossEntropy":
            delta_y = np.zeros(self.preds.shape)
            delta_y[range(bs), self.ys] = -1/(bs*self.preds[range(bs), self.ys])
        elif self.loss_type == "MSE":
            delta_y = 2*(self.preds-self.ys.reshape(-1, 1))/bs

        # loss's gradient to parameters of neural network
        self.delta_Ws = [None for i in range(self.nlayers)]
        self.delta_bs = [None for i in range(self.nlayers)]

        # loss's gradient to output of the last layer (after activation)
        delta_this_layer_out_act = delta_y

        # gradient backward propogation
        for layer_idx in range(self.nlayers-1, -1, -1):
            # inputs and outputs of this layer
            layer_in = self.xs if layer_idx == 0 else self.layer_outs[layer_idx-1]['AfterActivation']
            layer_out = self.layer_outs[layer_idx]['BeforeActivation']
            layer_out_act = self.layer_outs[layer_idx]['AfterActivation']

            # gradient of activation function
            actfunc_grads = [None for i in range(bs)]
            if self.activation_functions[layer_idx] == "softmax":
                for idx in range(bs):
                    actfunc_grads[idx] = -np.matmul(layer_out_act[idx].reshape(-1, 1), layer_out_act[idx].reshape(1, -1))
                    actfunc_grads[idx] += np.diag(layer_out_act[idx])
                actfunc_grads = np.array(actfunc_grads)

            elif self.activation_functions[layer_idx] == "relu":
                for idx in range(bs):
                    actfunc_grads[idx] = np.diag(np.where(layer_out_act[idx] > 0, 1, 0))
                actfunc_grads = np.array(actfunc_grads)

            elif self.activation_functions[layer_idx] == "sigmoid":
                for idx in range(bs):
                    actfunc_grads[idx] = np.diag(layer_out_act[idx]*(1-layer_out_act[idx]))
                actfunc_grads = np.array(actfunc_grads)

            elif self.activation_functions[layer_idx] == "tanh":
                for idx in range(bs):
                    actfunc_grads[idx] = np.diag(1-layer_out_act[idx]**2)
                actfunc_grads = np.array(actfunc_grads)

            # propagation through activation function
            delta_this_layer_out = np.einsum('ijk, ikl -> ijl', delta_this_layer_out_act.reshape(bs, 1, -1), actfunc_grads)

            # loss's gradient to parameters of this layer
            delta_this_layer_W = np.matmul(layer_in.T, delta_this_layer_out.reshape(bs, -1))
            delta_this_layer_b = np.sum(delta_this_layer_out.reshape(bs, -1), axis=0)
            self.delta_Ws[layer_idx] = delta_this_layer_W
            self.delta_bs[layer_idx] = delta_this_layer_b

            # regularization
            if self.regularization == "L1":
                self.delta_Ws[layer_idx] += self.lamb*np.sign(self.Ws[layer_idx])/bs
            elif self.regularization == "L2":
                self.delta_Ws[layer_idx] += self.lamb*2*self.Ws[layer_idx]/bs

            # loss's gradient to input of this layer
            delta_this_layer_in = np.matmul(delta_this_layer_out.reshape(bs, -1), self.Ws[layer_idx].T)
            # update: this layer's input is last layer's output(after activation)
            delta_this_layer_out_act = delta_this_layer_in.copy()

    def train(self, batch_xs, batch_ys):
        """ train NN to fit [train_xs, train_ys] """
        self.xs, self.ys = batch_xs, batch_ys
        self.preds = self.forward(self.xs)      # forward propagation
        self.loss_type = "CrossEntropy" if self.task_type == "Classification" else "MSE"
        self.loss = self._loss(self.preds, self.ys, loss_type=self.loss_type)
        self.backward()                         # backward propagation

        # update parameters of NN
        for layer_idx in range(self.nlayers):
            self.Ws[layer_idx] -= self.lr*self.delta_Ws[layer_idx]
            self.bs[layer_idx] -= self.lr*self.delta_bs[layer_idx]

    def lr_update(self):
        self.lr *= self.lr_decay
        self.lr = max(self.lr, self.lr_min)

    def visualize(self):
        """ visualize neural network """
        node_names = [None for dim in self.ndims]
        # node name for input layer
        node_names[0] = ["x[%d]"%(i) for i in range(self.ndims[0])]
        # node name for hidden layers
        for idx in range(1, self.nlayers):
            node_names[idx] = ["h%d[%d]"%(idx, i) for i in range(self.ndims[idx])]
        # node name for output layer
        node_names[-1] = ["y[%d]"%(i) for i in range(self.ndims[-1])]

        # plot neural network
        graph = Digraph("Neural Network", filename="NeuralNetwork")
        # input layer
        for name in node_names[0]:
            graph.node(name=name, label=name, fontsize='36', color="skyblue", style="filled")
        # hidden layers and output layer
        for layer_idx in range(self.nlayers):
            for node_id in range(self.ndims[layer_idx+1]):
                graph.node(name=node_names[layer_idx+1][node_id], label=node_names[layer_idx+1][node_id], fontsize='36', color="skyblue", style="filled")
                for forward_node_id in range(self.ndims[layer_idx]):
                    graph.edge(node_names[layer_idx][forward_node_id], node_names[layer_idx+1][node_id], \
                        label="%.2f"%self.Ws[layer_idx][forward_node_id][node_id], fontsize='12', style="filled")

        graph.view()