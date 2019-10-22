#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from eml.tree import describe

""" 
IMPORTANT 

In order to embed a decision tree model we must take into account 
the way the thresholds are testes by the model. 

Example:

    scikit learn tests the threshold for each attribute considering 

        th_1 <= x < th_2

    every time the equality will be tested on the "left" threshold.

    We might have some models, for example Weka, where the equality 
    test is made on the right side of the range. 

To make stuff more simple we established a convention for this model, 
where the equality test is always considered to be on the left side. 

This can be actually achived very easily using the machine epsilon,
upper bound on the relative error due to rounding in floating point 
arithmetic, by "shifting" the equality test from right to left summing
this value

Example

    let us consider a model where the equality test is made on the right-
    hand of the range:

        th_1 < x <= th_2

    we can change this as follows

        th_1 + epsilon < x <= th_2 - epsilon    <=>

        <=> th_1 <= x < th_2

This still needs to be implemented
"""


# give one of feature_list or feature_dict to map it to the names
def read_sklearn_tree(tree):
    """ Import decision tree model from sklearn

    Casts decision tree into custom representation, available at
    :obj:`eml.tree.describe.DTNode`

    Parameters
    ----------
        tree : :obj:`sklearn.tree.BaseDecisionTree`
            Trained decision tree

    Returns
    -------
        Decision tree : :obj:`eml.tree.describe.DTNode`
            Decision tree with custom representation

    """ 
    root = describe.DTNode()
    return _sklearn_tree_export(tree.tree_, 0, root)


def _sklearn_tree_export(tree, nid, root):
    """ COMMENT """
    # ------------------------------------------------------------------------
    # LEAF
    # ------------------------------------------------------------------------
    if tree.children_left[nid] < nid:
        if tree.n_outputs == 1:
            if tree.n_classes[0] == 1:  # regression
                val = tree.value[nid][0, 0]
            else:  # classification
                val = np.argmax(tree.value[nid])
        root.set_class(val)
        return root
    # ------------------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------------------
    # getting name, type and thresholds of the split
    attr_name = tree.feature[nid]
    attr_type = 0
    range_left = (-float('inf'), tree.threshold[nid])
    range_right = (tree.threshold[nid], float('inf'))
    # creating recursively the subtrees
    child_left = describe.DTNode(attr_name, attr_type, range_left)
    child_right = describe.DTNode(attr_name, attr_type, range_right)
    # attaching child
    root.add_child(child_left)
    root.add_child(child_right)
    # creating subtree
    subtree_left = _sklearn_tree_export(tree, tree.children_left[nid], child_left)
    subtree_right = _sklearn_tree_export(tree, tree.children_right[nid], child_right)

    return root
    