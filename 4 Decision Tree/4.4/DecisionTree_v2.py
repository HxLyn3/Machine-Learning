"""
- Author: Haoxin Lin
- E-mail: linhx36@outlook.com
- Date: 2020.10.19
- Brief: A Decision Tree class. You can select Information Gain or Gini Index to choose
optimal attribute for partition when building the Decision Tree. In addition, the function
of pre-pruning or post-pruning is provided.
"""

# -*- coding: utf-8 -*-

import re
import numpy as np
from graphviz import Digraph

class Node():
    """ Node class """

    def __init__(self, attr_name, isdisc):
        """ Init a node in the decision tree """
        self.isleaf = False             # not leaf node
        self.isvirtual = False          # not virtual node
        self.attr_name = attr_name      # attribute name
        self.isdisc = isdisc            # equals to 1 if the attribute is discrete, otherwise 0
        self.childs = []                # child nodes

    def insertChild(self, val, node):
        """ insert child node """
        if not self.isleaf:
            if self.isvirtual:
                self.childs[val] = node
            elif self.isdisc and val in self.attr_vals:
                self.childs[self.attr_vals.index(val)] = node
            elif not self.isdisc and (val == 0 or val == 1):
                self.childs[val] = node     # val = 0 means less than threshold, and val = 1 means greater
            else:
                print("Attribute value is error.")
        else:
            print("The node is leaf node.")

    def referChild(self, val):
        """ refer child node """
        v = re.compile(r'^[-+]?[0-9]+(\.[0-9]+)?$')         # float or int reg
        if not self.isleaf:
            if self.isvirtual:
                return self.childs[val]
            elif self.isdisc and val in self.attr_vals:
                return self.childs[self.attr_vals.index(val)]
            elif not self.isdisc and bool(v.match(str(val))):
                return self.childs[float(val)>self.threshold]
            else:
                print("Attribute value is error.")
        else:
            print("The node is leaf node.")
        return None

    def depth(self):
        """ return node's depth in the tree """
        if type(self) == LeafNode:
            return 0
        else:
            return max([child.depth() for child in self.childs]) + 1

    def getLabel(self):
        """ return label of leaf node """
        if self.isleaf:
            return self.label

class VirtualNode(Node):
    """ virtual node """
    def __init__(self):
        Node.__init__(self, attr_name=None, isdisc=None)
        self.isvirtual = True            # virtual node
        self.childs = [None]

class DiscreteNode(Node):
    """ Node of discrete attribute """
    def __init__(self, attr_name, attr_vals):
        Node.__init__(self, attr_name=attr_name, isdisc=True)
        self.attr_vals = attr_vals      # attribute values
        self.childs = [None]*len(self.attr_vals)

class ContinuousNode(Node):
    """ Node of continuous attribute """
    def __init__(self, attr_name, thres):
        Node.__init__(self, attr_name=attr_name, isdisc=False)
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

    def buildTree(self, train_xs, train_ys, test_xs, test_ys, attributes, isdiscs, labels, partIndex='InformationGain', prepruning=False):
        """ build decision tree """
        self.train_xs = train_xs
        self.train_ys = train_ys
        self.test_xs = test_xs
        self.test_ys = test_ys

        attr_values = [None]*len(attributes)
        for i in range(len(attributes)):
            if isdiscs[i]:
                attr_values[i] = list(set(np.concatenate((train_xs[:, i], test_xs[:, i]))))   # the set of values for each attributes
        self.attributes = attributes
        self.isdiscs = isdiscs
        self.labels = labels
        self.attr_values = attr_values
        self.partIndex = partIndex
        self.prepruning = prepruning

        self.handle = VirtualNode()
        self.buildTreeRecursive(self.handle, 0, self.train_xs, self.train_ys, partAttrs=attributes)
        self.root = self.handle.referChild(0)

    def buildTreeRecursive(self, father, branch_val, xs, ys, partAttrs):
        """ build decision tree recursively """

        # return leaf node when all the xs are in the same class
        if len(set(ys)) == 1:
            father.insertChild(branch_val, LeafNode(ys[0]))
            return

        # return leaf node labeled mode of ys when partAttrs is empty
        if len(partAttrs) == 0:
            father.insertChild(branch_val, LeafNode(np.argmax(np.bincount(ys))))
            return

        # return leaf node labeled mode of ys when values of xs in partAttrs are the same
        temp = 0
        for attr in partAttrs:
            idx = list(self.attributes).index(attr)
            temp += len(set(xs[:, idx]))
        if temp == len(partAttrs):
            father.insertChild(branch_val, LeafNode(np.argmax(np.bincount(ys))))
            return

        opt_attr = self.opt_partition(xs, ys, partAttrs, self.partIndex)
        opt_idx = list(self.attributes).index(opt_attr)     # index of optimal attribute in attributes
        isdisc = self.isdiscs[opt_idx]                      # whether the attribute is discrete

        # if the optimal attribute is discrete
        if isdisc:
            vals = self.attr_values[opt_idx]
            thisNode = DiscreteNode(opt_attr, vals)

            if self.prepruning:                             # pre-pruning
                # accuracy before partition
                father.insertChild(branch_val, LeafNode(np.argmax(np.bincount(ys))))
                beforePartAcc = self.test(self.test_xs, self.test_ys)
                # accuracy after partition
                father.insertChild(branch_val, thisNode)
                for val in vals:
                    sub_ys = ys[xs[:, opt_idx]==val]
                    if len(sub_ys) == 0:
                        thisNode.insertChild(val, LeafNode(np.argmax(np.bincount(ys))))
                    else:
                        thisNode.insertChild(val, LeafNode(np.argmax(np.bincount(sub_ys))))
                afterPartAcc = self.test(self.test_xs, self.test_ys)

                if beforePartAcc >= afterPartAcc:
                    father.insertChild(branch_val, LeafNode(np.argmax(np.bincount(ys))))
                    return

            father.insertChild(branch_val, thisNode)
            for val in vals:
                sub_xs = xs[xs[:, opt_idx]==val]
                sub_ys = ys[xs[:, opt_idx]==val]
                # if the dataset of the branch is empty
                if len(sub_ys) == 0:
                    thisNode.insertChild(val, LeafNode(np.argmax(np.bincount(ys))))
                else:
                    branch_partAttrs = list(partAttrs)
                    branch_partAttrs.remove(opt_attr)
                    self.buildTreeRecursive(thisNode, val, sub_xs, sub_ys, branch_partAttrs)
        # if the optimal attribute is continuous
        else:
            thres = self.contiThres[opt_attr]                # threshold of the continuous attribute
            thisNode = ContinuousNode(opt_attr, thres)

            if self.prepruning:                              # pre-pruning
                # accuracy before partition
                father.insertChild(branch_val, LeafNode(np.argmax(np.bincount(ys))))
                beforePartAcc = self.test(self.test_xs, self.test_ys)
                # accuracy after partition
                father.insertChild(branch_val, thisNode)
                ys_less = ys[xs[:, opt_idx].astype(np.float64)<=thres]
                if len(ys_less) == 0:
                    thisNode.insertChild(0, LeafNode(np.argmax(np.bincount(ys))))
                else:
                    thisNode.insertChild(0, LeafNode(np.argmax(np.bincount(ys_less))))
                ys_greater = ys[xs[:, opt_idx].astype(np.float64)>thres]
                if len(ys_greater) == 0:
                    thisNode.insertChild(1, LeafNode(np.argmax(np.bincount(ys))))
                else:
                    thisNode.insertChild(1, LeafNode(np.argmax(np.bincount(ys_greater))))
                afterPartAcc = self.test(self.test_xs, self.test_ys)

                if beforePartAcc >= afterPartAcc:
                    father.insertChild(branch_val, LeafNode(np.argmax(np.bincount(ys))))
                    return

            father.insertChild(branch_val, thisNode)
            # for branch that value of the attribute is less than threshold
            xs_less = xs[xs[:, opt_idx].astype(np.float64)<=thres]
            ys_less = ys[xs[:, opt_idx].astype(np.float64)<=thres]
            if len(ys_less) == 0:
                thisNode.insertChild(0, LeafNode(np.argmax(np.bincount(ys))))
            else:
                self.buildTreeRecursive(thisNode, 0, xs_less, ys_less, partAttrs)
            # for branch that value of the attribute is greater than threshold
            xs_greater = xs[xs[:, opt_idx].astype(np.float64)>thres]
            ys_greater = ys[xs[:, opt_idx].astype(np.float64)>thres]
            if len(ys_greater) == 0:
                thisNode.insertChild(1, LeafNode(np.argmax(np.bincount(ys))))
            else:
                self.buildTreeRecursive(thisNode, 1, xs_greater, ys_greater, partAttrs)

    def opt_partition(self, xs, ys, partAttrs, index='InformationGain'):
        self.contiThres = {}                        # threshold of continuous attributes

        # find the optimal attribute to partition the dataset
        if index == 'InformationGain':              # based on Information Gain
            prob = np.bincount(ys)/len(ys)
            prob = prob[prob!=0]
            Entropy = -sum(prob*np.log2(prob))      # entropy of dataset
            Gains = [Entropy]*len(partAttrs)        # information gains
            for i in range(len(partAttrs)):
                idx = list(self.attributes).index(partAttrs[i])
                # if the attribute is discrete
                if self.isdiscs[idx]:
                    valset = self.attr_values[idx]  # set of values for the attribute
                    for val in valset:
                        ys_v = ys[xs[:, idx]==val]
                        prob_v = np.bincount(ys_v)/len(ys_v)
                        prob_v = prob_v[prob_v!=0]
                        Ent_v = -sum(prob_v*np.log2(prob_v))    # entropy of divided dataset
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
                    self.contiThres[partAttrs[i]] = parts[k]
            opt_attr = partAttrs[np.argmax(Gains)]  # optimal attribute
        elif index == 'GiniIndex':                  # based on Gini Index
            Ginis = [0]*len(partAttrs)
            for i in range(len(partAttrs)):
                idx = list(self.attributes).index(partAttrs[i])
                # if the attribute is discrete
                if self.isdiscs[idx]:
                    valset = self.attr_values[idx]  # set of values for the attribute
                    for val in valset:
                        ys_v = ys[xs[:, idx]==val]
                        prob_v = np.bincount(ys_v)/len(ys_v)
                        Gini_v = 1 - sum(prob_v*prob_v)           # Gini Index of divided dataset
                        Ginis[i] += len(ys_v)/len(ys)*Gini_v    # update Gini Index
                # if the attribute is continuous
                else:
                    vals = sorted(xs[:, idx].astype(np.float64))
                    parts = [(vals[j]+vals[j+1])/2 for j in range(len(vals)-1)] # partitions
                    ginis = [0]*len(parts)
                    for k in range(len(parts)):
                        # data whose values in the attribute is less than partition
                        ys_less = ys[xs[:, idx].astype(np.float64)<=parts[k]]
                        prob_less = np.bincount(ys_less)/len(ys_less)
                        gini_less = 1 - sum(prob_less*prob_less)

                        ys_greater = ys[xs[:, idx].astype(np.float64)>parts[k]]
                        prob_greater = np.bincount(ys_greater)/len(ys_greater)
                        gini_greater = 1 - sum(prob_greater*prob_greater)
                        ginis[k] = (len(ys_less)*gini_less+len(ys_greater)*gini_greater)/len(ys)
                    k = np.argmin(ginis)
                    Ginis[i] += ginis[k]
                    self.contiThres[partAttrs[i]] = parts[k]
            opt_attr = partAttrs[np.argmin(Ginis)]  # optimal attribute
        return opt_attr

    def post_pruning(self):
        """ post pruning """
        self.post_pruning_recursive(self.train_xs, self.train_ys, self.handle, 0, self.root)

    def post_pruning_recursive(self, xs, ys, father, val, node):
        """ post pruning recursively """
        if type(node) == LeafNode:
            return True

        # for discrete node
        if type(node) == DiscreteNode:
            vals = node.attr_vals
            ndepths = [-child.depth() for child in node.childs]
            vals = np.array(vals)[np.argsort(ndepths)]
            prunings = [None]*len(vals)
            for i in range(len(vals)):
                sub_xs = xs[xs[:, list(self.attributes).index(node.attr_name)]==vals[i]]
                sub_ys = ys[xs[:, list(self.attributes).index(node.attr_name)]==vals[i]]
                prunings[i] = self.post_pruning_recursive(sub_xs, sub_ys, node, vals[i], node.referChild(vals[i]))

        # for continuous node
        elif type(node) == ContinuousNode:
            thres = node.threshold
            prunings = [None]*2
            ndepths = [-child.depth() for child in node.childs]
            for sign in np.argsort(ndepths):    # sign = 0: branch whose values less than thres
                if sign == 0:
                    sub_xs = xs[xs[:, list(self.attributes).index(node.attr_name)].astype(np.float64)<=thres]
                    sub_ys = ys[xs[:, list(self.attributes).index(node.attr_name)].astype(np.float64)<=thres]
                else:
                    sub_xs = xs[xs[:, list(self.attributes).index(node.attr_name)].astype(np.float64)>thres]
                    sub_ys = ys[xs[:, list(self.attributes).index(node.attr_name)].astype(np.float64)>thres]
                prunings[sign] = self.post_pruning_recursive(sub_xs, sub_ys, node, sign, node.referChild(sign))

        if sum(prunings) == 0:
            return False
        else:
            if len(ys) == 0:
                return False
            # accuracy before pruning
            beforePrunAcc = self.test(self.test_xs, self.test_ys)
            # accuracy after pruning
            father.insertChild(val, LeafNode(np.argmax(np.bincount(ys))))
            afterPrunAcc = self.test(self.test_xs, self.test_ys)
            if beforePrunAcc >= afterPrunAcc:
                father.insertChild(val, node)
                return False
            else:
                return True

    def visualize(self, graph_name):
        graph = Digraph(graph_name, filename=graph_name)
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
        tmp_node = self.handle.referChild(0)
        while type(tmp_node) != LeafNode:
            tmp_node = tmp_node.referChild(x[list(self.attributes).index(tmp_node.attr_name)])
        return tmp_node.label

    def test(self, test_xs, test_ys):
        """ test input dataset """
        preds = [self.classify(test_x) for test_x in test_xs]
        isCorrect = preds == test_ys
        accuracy = sum(isCorrect)/len(isCorrect)
        return accuracy