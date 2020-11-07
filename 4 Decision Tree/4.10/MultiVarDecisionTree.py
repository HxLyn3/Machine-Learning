"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.11.07
- Brief: A Multi-Variate Decision Tree class
"""

# -*- coding: utf-8 -*-

import re
import numpy as np
from graphviz import Digraph
from LogitReg import LogitReg

class Node():
    """ Node class """

    def __init__(self, attr_name, isdisc):
        """ Init a node in the decision tree """
        self.attr_name = attr_name      # attribute name
        self.isdisc = isdisc            # equals to 1 if the attribute is discrete, otherwise 0
        self.childs = []                # child nodes

    def insertChild(self, val, node):
        """ insert child node """
        if not self.isleaf:
            if self.isdisc and val in self.attr_vals:
                self.childs[self.attr_vals.index(val)] = node
            elif not self.isdisc and (val == 0 or val == 1):
                self.childs[val] = node     # val = 0 means less than threshold, and val = 1 means greater
            else:
                print("Attribute value is error.")
        else:
            print("The node is leaf node.")

    def referChild(self, val):
        """ refer child node """
        v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')     # float or int reg
        if not self.isleaf:
            if self.isdisc and val in self.attr_vals:
                return self.childs[self.attr_vals.index(val)]
            elif not self.isdisc and bool(v.match(str(val))):
                return self.childs[float(val)>self.threshold]
            else:
                print("Attribute value is error.")
        else:
            print("The node is leaf node.")
        return None

    def getLabel(self):
        """ return label of leaf node """
        if self.isleaf:
            return self.label

class DiscreteNode(Node):
    """ Node of discrete attribute """
    def __init__(self, attr_name, attr_vals):
        Node.__init__(self, attr_name=attr_name, isdisc=True)
        self.isleaf = False             # not leaf node
        self.attr_vals = attr_vals      # attribute values
        self.childs = [None]*len(self.attr_vals)

class ContinuousNode(Node):
    """ Node of continuous attribute """
    def __init__(self, attr_name, thres):
        Node.__init__(self, attr_name=attr_name, isdisc=False)
        self.isleaf = False             # not leaf node
        self.threshold = thres          # threshold of continuous values
        self.childs = [None]*2          # less and greater

class MultiVarNode(ContinuousNode):
    """ Multi-Variate Node """
    def __init__(self, attrs, weights, thres):
        self.attrs, self.weights = attrs, weights
        attr_name = "+".join(["%.3f×%s"%(self.weights[idx], self.attrs[idx]) for idx in range(len(self.attrs))])
        ContinuousNode.__init__(self, attr_name=attr_name, thres=thres)

class LeafNode(Node):
    """ Leaf Node """
    def __init__(self, label):
        Node.__init__(self, attr_name=None, isdisc=None)
        self.isleaf = True              # leaf node
        self.label = label              # label of the leaf node


class MultiVarDecisionTree():
    """ Multi-Variate Decision Tree """

    def __init__(self, train_xs, train_ys, attributes, labels):
        self.train_xs = train_xs
        self.train_ys = train_ys
        self.attributes = attributes
        self.labels = labels

    def buildTree(self):
        """ build decision tree """
        self.root = self.buildTreeRecursive(self.train_xs, self.train_ys, self.attributes)

    def buildTreeRecursive(self, xs, ys, attributes):
        """ build decision tree recursively """

        # return leaf node when all the xs are in the same class
        if len(set(ys)) == 1:
            return LeafNode(ys[0])

        # Logit Regression
        LogitRegressor = LogitReg(xs.shape[1])
        LogitRegressor.load(xs, ys)
        LogitRegressor.learn(steps=50000)
        preds = np.array([LogitRegressor.predict(x) for x in xs])
        thisNode = MultiVarNode(attributes, LogitRegressor.w, -LogitRegressor.b)

        # for branch that weighted sum of the attribute values is less than threshold
        xs_less = xs[preds==0]
        ys_less = ys[preds==0]
        thisNode.insertChild(0, self.buildTreeRecursive(xs_less, ys_less, attributes))
        # for branch that weighted sum of the attribute values is greater than threshold
        xs_greater = xs[preds==1]
        ys_greater = ys[preds==1]
        thisNode.insertChild(1, self.buildTreeRecursive(xs_greater, ys_greater, attributes))

        return thisNode

    def visualize(self):
        graph = Digraph("Decision Tree", filename="DecisionTree")
        self.nodeNameCnt = 0
        self.plotTreeRecursive(graph, self.root, father_name=None, branch_name=None)
        graph.view()

    def plotTreeRecursive(self, graph, cur_node, father_name, branch_name):
        """ plot tree recursively """
        cur_name = str(self.nodeNameCnt)
        self.nodeNameCnt += 1
        if type(cur_node) == LeafNode and cur_name == '0':
            graph.node(name=cur_name, label=self.labels[cur_node.label], \
                color="skyblue", fontname="FangSong", style='filled')
        elif type(cur_node) == LeafNode and cur_name != '0':
            graph.node(name=cur_name, label=self.labels[cur_node.label], \
                color="skyblue", fontname="FangSong", style='filled')
            graph.edge(father_name, cur_name, label=branch_name, fontname="FangSong", fontsize='12', style='filled')
        elif type(cur_node) == DiscreteNode:
            graph.node(name=cur_name, label=cur_node.attr_name+"=?", \
                shape='box', color="skyblue", fontname="FangSong", fontsize='12', style='filled')
            if father_name:
                graph.edge(father_name, cur_name, label=branch_name, fontname="FangSong", fontsize='12', style='filled')
            # child nodes
            father_name = cur_name
            for i in range(len(cur_node.childs)):
                self.plotTreeRecursive(graph, cur_node.childs[i], father_name, cur_node.attr_vals[i])
        elif type(cur_node) == ContinuousNode or type(cur_node) == MultiVarNode:
            graph.node(name=cur_name, label=cur_node.attr_name+"<="+"%.3f"%cur_node.threshold+"?", \
                shape='box', color="skyblue", fontname="FangSong", fontsize='12', style='filled')
            if father_name:
                graph.edge(father_name, cur_name, label=branch_name, fontname="FangSong", fontsize='12', style='filled')
            # child nodes
            father_name = cur_name
            self.plotTreeRecursive(graph, cur_node.childs[0], father_name, "是")
            self.plotTreeRecursive(graph, cur_node.childs[1], father_name, "否")

    def classify(self, x):
        """ classify x """
        tmp_node = self.root
        while type(tmp_node) != LeafNode:
            tmp_node = tmp_node.referChild(tmp_node.weights.dot(x))
        return tmp_node.label

    def test(self, test_xs, test_ys):
        """ test input dataset """
        preds = [self.classify(test_x) for test_x in test_xs]
        isCorrect = preds == test_ys
        accuracy = sum(isCorrect)/len(isCorrect)
        return accuracy