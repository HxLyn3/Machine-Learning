"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.10.12
- Brief: A Decision Tree class
"""

# -*- coding: utf-8 -*-

import re
import numpy as np
from graphviz import Digraph

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

class LeafNode(Node):
    """ Leaf Node """
    def __init__(self, label):
        Node.__init__(self, attr_name=None, isdisc=None)
        self.isleaf = True              # leaf node
        self.label = label              # label of the leaf node


class DecisionTree():
    """ Decision Tree """

    def __init__(self):
        self.root = None                # root node

    def buildTree(self, xs, ys, attributes, isdiscs, labels):
        """ build decision tree """
        attr_values = [None]*len(attributes)
        for i in range(len(attributes)):
            if isdiscs[i]:
                attr_values[i] = list(set(xs[:, i]))        # the set of values for each attributes
        self.root = self.buildTreeRecursive(xs, ys, attributes, isdiscs, attr_values, partAttrs=attributes)

        self.train_xs = xs
        self.train_ys = ys
        self.attributes = attributes
        self.labels = labels

    def buildTreeRecursive(self, xs, ys, attributes, isdiscs, attr_values, partAttrs):
        """ build decision tree recursively """

        # return leaf node when all the xs are in the same class
        if len(set(ys)) == 1:
            return LeafNode(ys[0])

        # return leaf node labeled mode of ys when partAttrs is empty
        if len(partAttrs) == 0:
            return LeafNode(np.argmax(np.bincount(ys)))

        # return leaf node labeled mode of ys when values of xs in partAttrs are the same
        temp = 0
        for attr in partAttrs:
            idx = list(attributes).index(attr)
            temp += len(set(xs[:, idx]))
        if temp == len(partAttrs):
            return LeafNode(np.argmax(np.bincount(ys)))

        contiThres = {}                      # threshold of discrete attributes

        # find the optimal attribute to partition the dataset
        prob = np.bincount(ys)/len(ys)
        prob = prob[prob!=0]
        Entropy = -sum(prob*np.log2(prob))  # entropy of dataset
        Gains = [Entropy]*len(partAttrs)    # information gains
        for i in range(len(partAttrs)):
            idx = list(attributes).index(partAttrs[i])
            # if the attribute is discrete
            if isdiscs[idx]:
                valset = attr_values[idx]   # set of values for the attribute
                for val in valset:
                    ys_v = ys[xs[:, idx]==val]
                    prob_v = np.bincount(ys_v)/len(ys_v)
                    prob_v = prob_v[prob_v!=0]
                    Ent_v = -sum(prob_v*np.log2(prob_v))   # entropy of divided dataset
                    Gains[i] -= len(ys_v)/len(ys)*Ent_v     # update information gain
            # if the attribute is continuous
            else:
                vals = sorted(xs[:, idx].astype(np.float64))
                parts = [(vals[j]+vals[j+1])/2 for j in range(len(vals)-1)] # partitions
                ents = [0]*len(parts)
                for k in range(len(parts)):
                    # data whose values in the attribute is less than partition
                    ys_less = ys[xs[:, idx].astype(np.float64)<=parts[k]]
                    prob_less = np.bincount(ys_less)/len(ys_less)
                    prob_less = prob_less[prob_less!=0]
                    Ent_less = -sum(prob_less*np.log2(prob_less))

                    ys_greater = ys[xs[:, idx].astype(np.float64)>parts[k]]
                    prob_greater = np.bincount(ys_greater)/len(ys_greater)
                    prob_greater = prob_greater[prob_greater!=0]
                    Ent_greater = -sum(prob_greater*np.log2(prob_greater))
                    ents[k] = (len(ys_less)*Ent_less+len(ys_greater)*Ent_greater)/len(ys)
                k = np.argmin(ents)
                Gains[i] -= ents[k]
                contiThres[partAttrs[i]] = parts[k]
        opt_attr = partAttrs[np.argmax(Gains)]  # optimal attribute
        opt_idx = list(attributes).index(opt_attr)    # index of optimal attribute in attributes
        isdisc = isdiscs[opt_idx]               # whether the attribute is discrete

        # if the optimal attribute is discrete
        if isdisc:
            vals = attr_values[opt_idx]
            thisNode = DiscreteNode(opt_attr, vals)
            for val in vals:
                sub_xs = xs[xs[:, opt_idx]==val]
                sub_ys = ys[xs[:, opt_idx]==val]
                # if the dataset of the branch is empty
                if len(sub_ys) == 0:
                    thisNode.insertChild(val, LeafNode(np.argmax(np.bincount(ys))))
                else:
                    branch_partAttrs = list(partAttrs)
                    branch_partAttrs.remove(opt_attr)
                    thisNode.insertChild(val, self.buildTreeRecursive(sub_xs, sub_ys, attributes, isdiscs, attr_values, branch_partAttrs))
        # if the optimal attribute is continuous
        else:
            thres = contiThres[opt_attr]        # threshold of the continuous attribute
            thisNode = ContinuousNode(opt_attr, thres)
            # for branch that value of the attribute is less than threshold
            xs_less = xs[xs[:, opt_idx].astype(np.float64)<=thres]
            ys_less = ys[xs[:, opt_idx].astype(np.float64)<=thres]
            if len(ys_less) == 0:
                thisNode.insertChild(0, LeafNode(np.argmax(np.bincount(ys))))
            else:
                thisNode.insertChild(0, self.buildTreeRecursive(xs_less, ys_less, attributes, isdiscs, attr_values, partAttrs))
            # for branch that value of the attribute is greater than threshold
            xs_greater = xs[xs[:, opt_idx].astype(np.float64)>thres]
            ys_greater = ys[xs[:, opt_idx].astype(np.float64)>thres]
            if len(ys_greater) == 0:
                thisNode.insertChild(1, LeafNode(np.argmax(np.bincount(ys))))
            else:
                thisNode.insertChild(1, self.buildTreeRecursive(xs_greater, ys_greater, attributes, isdiscs, attr_values, partAttrs))

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
        if type(cur_node) == LeafNode:
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
        elif type(cur_node) == ContinuousNode:
            graph.node(name=cur_name, label=cur_node.attr_name+"<="+str(cur_node.threshold)+"?", \
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
            tmp_node = tmp_node.referChild(x[list(self.attributes).index(tmp_node.attr_name)])
        return tmp_node.label

    def test(self, test_xs, test_ys):
        """ test input dataset """
        preds = [self.classify(test_x) for test_x in test_xs]
        isCorrect = preds == test_ys
        accuracy = sum(isCorrect)/len(isCorrect)
        return accuracy