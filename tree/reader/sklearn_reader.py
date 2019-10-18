#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from eml.tree import describe


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
    return _sklearn_tree_export(tree.tree_, 0, None)

def _sklearn_tree_export(tree, nid, dtParent):
    # TODO
    # Build a node object
    root = None if dtParent is None else dtParent._root
    dtMe = describe.DTNode(root)
    # ------------------------------------------------------------------------
    # if is leaf
    if tree.children_left[nid] < nid:
        # ... and tree.children_left[node] < node:
        if tree.n_outputs == 1:
            if tree.n_classes[0] == 1:
                # regression
                val = tree.value[nid][0, 0]
            else: # classification
                val = np.argmax(tree.value[nid])
        dtMe.set_class_label(str(val))
        # XXX aname, atype?
        return dtMe
    # ------------------------------------------------------------------------
    # If there are children, configure the branch related fields
    aname = tree.feature[nid] #str(tree.feature[nid])
    # atype = describe.DTNode.attr_num
    atype = 0
    label_left = (-float('inf'), tree.threshold[nid])
    label_right = (tree.threshold[nid], float('inf'))
    # Then process the children
    dt_left = _sklearn_tree_export(tree, tree.children_left[nid], dtMe)
    dt_right = _sklearn_tree_export(tree, tree.children_right[nid], dtMe)
    # Finally, attach the converted children to the current node
    dtMe.add_child(dt_left, aname, atype, label_left)
    dtMe.add_child(dt_right, aname, atype, label_right)
    # ------------------------------------------------------------------------
    return dtMe
